import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# パラメータ設定 (ご提示のコードと同じ)
# --------------------------------------------------
a = 0.011
b = 0.016
f_star = 0.5109
V_star = 1e-4      # m/s
D_RS = 1e-6        # m

# ご提示いただいた関数 generate_rs_data をそのまま利用
# def generate_rs_data(V_func, t_eval): ...


# --------------------------------------------------
# 速度軌道 V(t) をランダム生成するヘルパー関数
# --------------------------------------------------

def create_velocity_jump_func(T=1.98, log_v_min=-3.0, log_v_max=1.0, num_jumps=3):
    """① 速度跳躍 (Velocity Jump) 軌道を生成"""
    # 0~Tの間でランダムなジャンプ時刻を設定
    jump_times = sorted(np.random.uniform(0.1, T - 0.1, size=num_jumps))
    jump_times = [0.0] + jump_times + [T + 0.1]
    
    # 各区間の log10(V/V*) をランダム設定
    log_v_values = np.random.uniform(log_v_min, log_v_max, size=num_jumps + 1)
    
    def V_func(t):
        idx = np.searchsorted(jump_times, t, side='right') - 1
        idx = np.clip(idx, 0, len(log_v_values) - 1)
        return V_star * (10 ** log_v_values[idx])
        
    return V_func

def create_continuous_variation_func(T=1.98, log_v_min=-3.0, log_v_max=1.0, num_control_points=5):
    """② 連続変化 (Continuous Variation) 軌道を生成"""
    # 時間軸上のいくつかの点でランダムな log10(V/V*) を設定
    t_points = np.linspace(0, T, num_control_points)
    log_v_values = np.random.uniform(log_v_min, log_v_max, size=num_control_points)
    
    # スプライン補間（または線形補間）で連続化
    interp_func = interp1d(t_points, log_v_values, kind='cubic', fill_value="extrapolate")
    
    def V_func(t):
        log_v = np.clip(interp_func(t), log_v_min, log_v_max)
        return V_star * (10 ** log_v)
        
    return V_func


# --------------------------------------------------
# 800個の正解データ生成ループ
# --------------------------------------------------

# 時間軸: 100点 (dt = 0.02秒) -> 0.00, 0.02, ..., 1.98 秒
t_eval = np.arange(100) * 0.02  

X_list = []  # 入力: 速度軌道 V(t)
Y_list = []  # 出力 (正解): 摩擦係数 f_RS(t)

num_samples = 800

for i in range(num_samples):
    # 半分は速度跳躍、半分は連続変化を選択
    if i % 2 == 0:
        V_func = create_velocity_jump_func(T=t_eval[-1])
    else:
        V_func = create_continuous_variation_func(T=t_eval[-1])
        
    # 微分方程式を解いて V(t) と f_RS(t) を取得
    V_sol, f_RS = generate_rs_data(V_func, t_eval)
    
    X_list.append(V_sol)
    Y_list.append(f_RS)

# NumPy 配列化 (Shape: (800, 100))
X_data = np.array(X_list)
Y_data = np.array(Y_list)

# --------------------------------------------------
# 論文通りの比率でデータ分割 (Train 480 / Val 160 / Test 160)
# --------------------------------------------------
# 800個のうち 480個(60%) を訓練用、残り320個(40%) に分割
X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.4, random_state=42)

# 残り320個を半分ずつ(160個/160個)に分割
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"訓練データ   (Train): X={X_train.shape}, Y={Y_train.shape}") # (480, 100)
print(f"検証データ   (Val)  : X={X_val.shape}, Y={Y_val.shape}")     # (160, 100)
print(f"テストデータ (Test) : X={X_test.shape}, Y={Y_test.shape}")    # (160, 100)