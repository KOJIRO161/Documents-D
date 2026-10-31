#!/usr/bin/env python3
"""Small-test training of the potential RNO in Liu et al. (2025).

Dataset sizes: train=3, validation=1, test=1; each history has 100 samples.
The code follows Algorithm 1 with explicit-Euler internal-state updates and
reverse-mode differentiation through both neural potentials.
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

# ---------- Configuration ----------
@dataclass
class Config:
    seed: int = 2025
    n_train: int = 3
    n_val: int = 1
    n_test: int = 1
    n_time: int = 100
    dt: float = 0.02
    a: float = 0.011
    b: float = 0.016
    f_ref: float = 0.5109
    v_ref: float = 1.0e-4
    d_rs: float = 1.0e-6
    xi_dim: int = 1
    hidden_dagger: tuple = (64, 64)
    hidden_star: tuple = (64, 64)
    epochs: int = 1000
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-8
    grad_clip: float = 10.0
    patience: int = 150
    dtype: str = "float64"


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ---------- Synthetic RSF data ----------
def make_velocity_ratio(rng, n=100):
    """50:50 jump or smooth log10(V/Vref) history in [1e-3, 1e1]."""
    if rng.random() < 0.5:
        nj = int(rng.integers(2, 6))
        cuts = np.sort(rng.choice(np.arange(8, n-8), nj, replace=False))
        levels = rng.uniform(-3.0, 1.0, nj+1)
        y = np.full(n, levels[0])
        for k, lev in zip(cuts, levels[1:]): y[k:] = lev
        kind = "jump"
    else:
        knots = np.linspace(0, n-1, 7).astype(int)
        vals = rng.uniform(-3.0, 1.0, len(knots))
        y = np.interp(np.arange(n), knots, vals)
        kind = "smooth"
    return 10.0**y, kind


def rsf_response(v, c: Config):
    """Aging-law response. Exact update for piecewise-constant V over each dt."""
    th = np.empty_like(v)
    th[0] = c.d_rs / v[0]  # initial steady state, explicit assumption
    for n in range(1, len(v)):
        ve = max(v[n-1], 1e-30)
        ss = c.d_rs / ve
        th[n] = ss + (th[n-1] - ss) * np.exp(-ve*c.dt/c.d_rs)
    f = c.f_ref + c.a*np.log(v/c.v_ref) + c.b*np.log(c.v_ref*th/c.d_rs)
    return th, f


def make_dataset(c):
    rng = np.random.default_rng(c.seed)
    out = {}
    for split, ns in (("train",c.n_train),("val",c.n_val),("test",c.n_test)):
        V=[]; F=[]; TH=[]; kinds=[]
        for _ in range(ns):
            ratio, kind = make_velocity_ratio(rng, c.n_time)
            v = c.v_ref*ratio
            th, f = rsf_response(v,c)
            V.append(v); TH.append(th); F.append(f); kinds.append(kind)
        out[split] = dict(V=np.asarray(V), theta=np.asarray(TH), f=np.asarray(F), kind=kinds)
    return out

# ---------- Potentials ----------
class MLP(nn.Module):
    def __init__(self, nin, hidden, nout=1):
        super().__init__()
        dims=(nin,)+tuple(hidden)+(nout,)
        layers=[]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i],dims[i+1]), nn.ELU()] # exponential ReLU
        layers += [nn.Linear(dims[-2],dims[-1])]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class PotentialRNO(nn.Module):
    """D^dagger(log(V/V*),xi), D*(d), and one learned xi_0 per training history."""
    def __init__(self,c):
        super().__init__(); self.c=c
        self.Ddagger=MLP(1+c.xi_dim,c.hidden_dagger,1)
        self.Dstar=MLP(c.xi_dim,c.hidden_star,1)
        self.xi0_train=nn.Parameter(torch.zeros(c.n_train,c.xi_dim))
        self.xi0_shared=nn.Parameter(torch.zeros(c.xi_dim))

    @staticmethod
    def grad_scalar(y,x):
        return torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]

    def rollout(self,V,xi0,create_graph=True):
        """Algorithm 1: Euler update xi, evaluate f_NN at all 100 instants."""
        B,T=V.shape; xi=xi0
        fs=[]; xis=[xi]
        for n in range(T):
            # u is dimensionless; V=Vref*exp(u). Ddagger=Vref*psi implies
            # dDdagger/dV = exp(-u)*dpsi/du.
            u=torch.log(V[:,n:n+1]/self.c.v_ref).requires_grad_(True)
            xi=xi.requires_grad_(True)
            psi=self.Ddagger(torch.cat([u,xi],dim=1))
            dpsi_du=self.grad_scalar(psi,u)
            dpsi_dxi=self.grad_scalar(psi,xi)
            # Analytic direct-effect potential derivative + learned state term.
            f_direct=self.c.f_ref + self.c.a*u
            f_state=torch.exp(-u)*dpsi_du
            fs.append((f_direct+f_state).squeeze(1))
            if n < T-1:
                driving=-dpsi_dxi
                Dstar=self.Dstar(driving)
                xidot=self.grad_scalar(Dstar,driving)
                xi=xi+self.c.dt*xidot
                xis.append(xi)
        return torch.stack(fs,1), torch.stack(xis,1)


def relative_l2(pred,target):
    return torch.linalg.vector_norm(pred-target)/torch.linalg.vector_norm(target)


def initial_xi(model,split,batch_size):
    if split=="train": return model.xi0_train[:batch_size]
    return model.xi0_shared.expand(batch_size,-1)

@torch.enable_grad()
def evaluate(model,V,F,split):
    pred,_=model.rollout(V,initial_xi(model,split,V.shape[0]))
    return float(relative_l2(pred,F).detach()), pred.detach()


def train(c, outdir):
    seed_all(c.seed)
    dtype=torch.float64 if c.dtype=="float64" else torch.float32
    torch.set_default_dtype(dtype)
    data=make_dataset(c)
    ten={s:{k:torch.as_tensor(d[k],dtype=dtype) for k in ("V","f")} for s,d in data.items()}
    model=PotentialRNO(c)
    opt=torch.optim.Adam(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=40,min_lr=1e-6)
    best=math.inf; best_state=None; stale=0; history=[]

    # With only 3 histories, one full batch is less noisy and exactly uses all data.
    for epoch in range(1,c.epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        pred,_=model.rollout(ten["train"]["V"],model.xi0_train)
        loss=relative_l2(pred,ten["train"]["f"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),c.grad_clip)
        opt.step()
        model.eval()
        val,_=evaluate(model,ten["val"]["V"],ten["val"]["f"],"val")
        scheduler.step(val)
        tr=float(loss.detach()); history.append((epoch,tr,val,opt.param_groups[0]["lr"]))
        if val < best:
            best=val; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if epoch==1 or epoch%50==0: print(f"epoch={epoch:4d} train={tr:.4e} val={val:.4e}")
        if stale>=c.patience: print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(best_state)
    metrics={}
    predictions={}
    for s in ("train","val","test"):
        err,p=evaluate(model,ten[s]["V"],ten[s]["f"],s)
        metrics[f"{s}_relative_l2"]=err; predictions[s]=p.cpu().numpy()
    outdir.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":asdict(c),"metrics":metrics},outdir/"rno_small_best.pt")
    np.savez_compressed(outdir/"rno_small_results.npz",
        history=np.asarray(history),
        **{f"{s}_{k}":v for s,d in data.items() for k,v in d.items() if k!="kind"},
        **{f"{s}_f_pred":v for s,v in predictions.items()})
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=1000)
    p.add_argument("--outdir",type=Path,default=Path("rno_small_output"))
    p.add_argument("--paper-widths",action="store_true",help="Use paper-reported large hidden widths; expensive for higher derivatives.")
    a=p.parse_args(); c=Config(epochs=a.epochs)
    if a.paper_widths:
        c.hidden_dagger=(1024,1024,1024,1024,64)
        c.hidden_star=(512,512,128,128,16,1024)
    train(c,a.outdir)

if __name__=="__main__": main()
