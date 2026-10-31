import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from torch.utils.data import TensorDataset, DataLoader

# 乱数 Seed の固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 物性パラメータ (Liu et al., 2026)
a = 0.011
b = 0.016
f_star = 0.5109
V_star = 1e-4
D_RS = 1e-6

# 無次元時間設定: t' = t * V_star / D_RS (0 ~ 20.0, 200ステップ)
t_dim_max = 20.0
n_steps = 200
t_eval_dim = np.linspace(0, t_dim_max, n_steps)
dt_dim = t_eval_dim[1] - t_eval_dim[0]

def generate_nondim_dataset(num_sequences=500, seed=42):
    np.random.seed(seed)
    V_batches, f_batches = [], []
    
    for seq_idx in range(num_sequences):
        seq_type = seq_idx % 2
        
        if seq_type == 0:
            # 1. 速度ジャンプ系列 (1~3回の不連続跳ね上がり)
            n_jumps = np.random.randint(1, 4)
            jumps = np.random.uniform(-1.0, 0.5, size=n_jumps)
            times = np.sort(np.random.uniform(2.0, 18.0, size=n_jumps-1))
            times = np.concatenate([[0.0], times, [t_dim_max]])
            
            def V_ratio_func(t_dim):
                for i in range(len(jumps)):
                    if times[i] <= t_dim < times[i+1]:
                        return 10 ** jumps[i]
                return 10 ** jumps[-1]
        else:
            # 2. Continuous variation (3次スプライン補間による滑らかな変化)
            t_knots = np.linspace(0, t_dim_max, 6)
            v_knots = np.random.uniform(-1.0, 0.5, size=6)
            cs = CubicSpline(t_knots, v_knots, bc_type='natural')
            
            def V_ratio_func(t_dim):
                log_v = float(cs(t_dim))
                return 10 ** log_v

        # 無次元状態方程式: d(theta')/dt' = 1 - V' * theta'
        def theta_dot(t_dim, theta_dim):
            v_r = V_ratio_func(t_dim)
            return 1.0 - v_r * theta_dim[0]

        # 初期条件: 初期速度に対する定常状態 theta_0' = 1 / V_0'
        v0_ratio = V_ratio_func(0)
        theta0_dim = [1.0 / v0_ratio]
        
        sol = solve_ivp(theta_dot, (0, t_dim_max), theta0_dim, t_eval=t_eval_dim, method='RK45', rtol=1e-8)
        theta_sol_dim = np.maximum(sol.y[0], 1e-15)
        
        V_ratio_sol = np.array([V_ratio_func(t) for t in t_eval_dim])
        f_RS = f_star + a * np.log(V_ratio_sol) + b * np.log(theta_sol_dim)
        
        V_batches.append(np.log10(V_ratio_sol))
        f_batches.append(f_RS)
        
    return torch.tensor(np.array(V_batches), dtype=torch.float32).to(device), \
           torch.tensor(np.array(f_batches), dtype=torch.float32).to(device)

# データ生成と 70:15:15 分割 (Train: 350, Val: 75, Test: 75)
V_all, f_all = generate_nondim_dataset(num_sequences=500, seed=SEED)

n_train, n_val = 350, 75
V_train, f_train = V_all[:n_train], f_all[:n_train]
V_val, f_val = V_all[n_train:n_train+n_val], f_all[n_train:n_train+n_val]
V_test, f_test = V_all[n_train+n_val:], f_all[n_train+n_val:]

print(f"データセット準備完了:")
print(f"  Train : {V_train.shape}")
print(f"  Val   : {V_val.shape}")
print(f"  Test  : {V_test.shape}")


########################


class PaperPotentialRNO(nn.Module):
    def __init__(self, dim_xi=1, hidden_dim=128):
        super().__init__()
        self.dim_xi = dim_xi
        
        # D_dagger MLP (散逸ポテンシャルの残差成分)
        self.net_D_dagger = nn.Sequential(
            nn.Linear(1 + dim_xi, hidden_dim), nn.CELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.CELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.CELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # D_star MLP (双対散逸ポテンシャル)
        self.net_D_star = nn.Sequential(
            nn.Linear(dim_xi, hidden_dim), nn.CELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.CELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_sequence(self, log_v_seq, dt_dim):
        batch_size, seq_len = log_v_seq.shape
        # 初期内部変数 xi_0 = 0
        xi = torch.zeros(batch_size, self.dim_xi, device=log_v_seq.device)
        
        f_NN_list = []
        for n in range(seq_len):
            log_v = log_v_seq[:, n:n+1]
            log_v_input = log_v.clone().requires_grad_(True)
            
            inputs_dagger = torch.cat([log_v_input, xi], dim=-1)
            mlp_out = self.net_D_dagger(inputs_dagger)
            
            # 物理インフォームド・事前知識ポテンシャル
            # 1/2 * a_eff * (log10(V/V*))^2 (※ log10 と ln のスケール変換補正)
            a_eff = a * np.log(10)
            base_potential = 0.5 * a_eff * (log_v_input ** 2)
            
            D_dagger = base_potential + mlp_out
            
            # 自動微分による勾配算定: [dD/d(log_v), dD/d(xi)]
            grads = torch.autograd.grad(D_dagger.sum(), inputs_dagger, create_graph=True)[0]
            grad_log_v = grads[:, 0:1]
            grad_xi = grads[:, 1:]
            
            f_NN = f_star + grad_log_v
            f_NN_list.append(f_NN)
            
            # 内部変数の時間発展 (双対ポテンシャル D* の微分)
            d = -grad_xi
            D_star = self.net_D_star(d)
            xi_dot = torch.autograd.grad(D_star.sum(), d, create_graph=True)[0]
            
            xi = xi + dt_dim * xi_dot
            
        return torch.cat(f_NN_list, dim=1)




###########################################################




batch_size = 32
train_loader = DataLoader(TensorDataset(V_train, f_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(V_val, f_val), batch_size=batch_size, shuffle=False)

model = PaperPotentialRNO(dim_xi=1, hidden_dim=128).to(device)

# オプティマイザとスケジューラの設定
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

print("モデルの学習・検証を開始します...")

for epoch in range(1, epochs + 1):
    # 訓練ステップ
    model.train()
    train_loss = 0.0
    
    for batch_V, batch_f in train_loader:
        optimizer.zero_grad()
        f_pred = model.forward_sequence(batch_V, dt_dim)
        
        loss = torch.mean((f_pred - batch_f) ** 2)
        loss.backward()
        
        # 勾配クリッピング（AutoGradの安定化）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * len(batch_V)
        
    scheduler.step()
    train_loss /= len(V_train)
    
    # 検証ステップ
    model.eval()
    val_loss = 0.0
    with torch.enable_grad():  # RNOのフォワードにAutoGradが必要なためenable_grad
        for batch_V_val, batch_f_val in val_loader:
            f_pred_val = model.forward_sequence(batch_V_val, dt_dim)
            v_loss = torch.mean((f_pred_val - batch_f_val) ** 2)
            val_loss += v_loss.item() * len(batch_V_val)
            
    val_loss /= len(V_val)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | Train MSE: {train_loss:.8f} | Val MSE: {val_loss:.8f}")

print("学習完了！")




###################################################



model.eval()
test_idx = 0  # テストデータのサンプル番号

with torch.enable_grad():
    f_pred_tensor = model.forward_sequence(V_test[test_idx:test_idx+1], dt_dim)

f_pred = f_pred_tensor.detach().cpu().numpy()[0]
f_target = f_test[test_idx].cpu().numpy()
v_input = V_test[test_idx].cpu().numpy()

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

# 描画 (Raw string r'' を用いて SyntaxWarning を回避)
ax1.plot(t_eval_dim, v_input, 'g:', label=r'$\log_{10}(V/V_*)$ (Input)')
ax2.plot(t_eval_dim, f_target, 'k--', label='Empirical RS ($f^{RS}$ Target)', linewidth=2)
ax2.plot(t_eval_dim, f_pred, 'r-', label='Paper Potential RNO ($f^{NN}$)', linewidth=1.5)

ax1.set_xlabel('Dimensionless Time $t\'$')
ax1.set_ylabel(r'$\log_{10}(V/V_*)$', color='g')
ax2.set_ylabel('Friction Coefficient $f$')
plt.title(f'Test Sample #{test_idx} Evaluation')
ax2.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()