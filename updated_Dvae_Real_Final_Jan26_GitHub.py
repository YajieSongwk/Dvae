#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 12:32:52 2026

@author: wukong
"""

import os
os.getcwd()
os.chdir('')
os.getcwd()



import os, random
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---- R bindings ----
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

CDM  = importr('CDM')    
NPCD = importr('NPCD')   # AlphaNP


# 0) Reproducibility

SEED = 43

def set_global_seeds(seed=SEED):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ro.r(f"set.seed({int(seed)})")

set_global_seeds(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Load from CSV

def load_from_csv(data_path, q_path):
    X = pd.read_csv(data_path).to_numpy().astype(int)   # N x J
    Q = pd.read_csv(q_path).to_numpy().astype(int)      # J x K
    return X, Q


# 2) Split + masking utils

def split_students(X, train=0.7, val=0.15, test=0.15, seed=SEED):
    idx = np.arange(X.shape[0])
    idx_tr, idx_tmp = train_test_split(idx, test_size=(1-train), random_state=seed, shuffle=True)
    rel = test / (val + test)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=rel, random_state=seed+1, shuffle=True)
    return idx_tr, idx_val, idx_te

def make_masks(X, prop=0.30, seed=SEED):
    rng = np.random.default_rng(seed)
    n, j = X.shape
    m = np.zeros((n, j), dtype=bool)
    k = max(1, int(round(prop * j)))
    for i in range(n):
        idx = rng.choice(j, size=k, replace=False)
        m[i, idx] = True  # True = UNSEEN
    return m

def build_known_matrix_python(X_full, mask, fill_value=0.5):
    X_known = X_full.astype(float).copy()
    X_known[mask] = float(fill_value)
    return X_known

def build_known_matrix_r(X_full, mask):
    X_known = X_full.astype(float).copy()
    X_known[mask] = np.nan  # R NA
    return X_known


# 3) Metrics on masked (unseen) items

def evaluate_on_unseen(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5) -> Dict[str, float]:
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    eps = 1e-7
    y_prob_clip = np.clip(y_prob, eps, 1 - eps)
    bce = -(y_true * np.log(y_prob_clip) + (1 - y_true) * np.log(1 - y_prob_clip)).mean()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    y_pred = (y_prob >= thr).astype(int)
    acc = (y_pred == y_true).mean()
    return {"BCE": float(bce), "AUC": float(auc), "ACC": float(acc), "ERROR": float(1.0 - acc)}


# 4) DVAE (SIMULATION-STYLE) + training + tuning

def sample_concrete_bernoulli(q, temperature=0.67, eps=1e-7):
    q = q.clamp(eps, 1.0 - eps)
    logit_q = torch.log(q) - torch.log(1.0 - q)
    u = torch.rand_like(q).clamp(eps, 1.0 - eps)
    g = torch.log(u) - torch.log(1.0 - u)
    z = torch.sigmoid((logit_q + g) / float(temperature))
    return z

def kl_bernoulli_freebits(q, prior_p=0.5, free_bits=0.02, eps=1e-7):
    
    q = q.clamp(min=eps, max=1.0 - eps)
    p = float(prior_p)

    kl_dim = (
        q * torch.log((q + eps) / p) +
        (1 - q) * torch.log((1 - q + eps) / (1 - p))
    )  # [N,K]

    kl_sum = kl_dim.sum(dim=1)  # [N]
    if free_bits is not None and free_bits > 0:
        K = q.shape[1]
        kl_sum = torch.clamp(kl_sum, min=float(free_bits) * K)

    return kl_sum.mean()


class DVAECDM(nn.Module):
    """
    Encoder: X -> q(Z=1|X)
    Sample:  z ~ Concrete(q, temp)
    Decoder: mixture-of-gates (AND/OR) with per-item gate pi_j,
             Q-masked nonnegative W2, item discrimination a_j, intercept b_j,
             explicit slip/guess constrained to [min,max].
    """
    def __init__(
        self,
        num_items,
        num_attrs,
        Q,
        hidden_units=64,
        dropout_rate=0.0,
        g_min=1e-4, g_max=0.499,
        s_min=1e-4, s_max=0.499,
        tau=5.0,
        init_gate=0.5
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.num_attrs = int(num_attrs)

        Q_tensor = torch.tensor(np.asarray(Q), dtype=torch.float32)  # [J,K]
        self.register_buffer("Q_JK", Q_tensor)                       # [J,K]
        self.register_buffer("Q_KJ", Q_tensor.t().contiguous())       # [K,J]

        # Encoder
        self.enc_hidden  = nn.Linear(self.num_items, hidden_units)
        self.enc_dropout = nn.Dropout(dropout_rate)
        self.enc_out     = nn.Linear(hidden_units, self.num_attrs)

        # Decoder
        self.W2_raw = nn.Parameter(torch.zeros(self.num_items, self.num_attrs))  # [J,K]
        self.a_raw  = nn.Parameter(torch.zeros(self.num_items))                  # [J]
        self.b_j    = nn.Parameter(torch.zeros(self.num_items))                  # [J]

        init_p = 0.15
        init_logit = torch.logit(torch.full((self.num_items,), float(init_p)))
        self.guess_logit = nn.Parameter(init_logit.clone())
        self.slip_logit  = nn.Parameter(init_logit.clone())
        self.g_min, self.g_max = float(g_min), float(g_max)
        self.s_min, self.s_max = float(s_min), float(s_max)

        gate_init_logit = torch.logit(torch.full((self.num_items,), float(init_gate)))
        self.gate_logit = nn.Parameter(gate_init_logit.clone())

        self.tau = float(tau)

        nn.init.xavier_uniform_(self.enc_hidden.weight); nn.init.zeros_(self.enc_hidden.bias)
        nn.init.xavier_uniform_(self.enc_out.weight);    nn.init.zeros_(self.enc_out.bias)

        with torch.no_grad():
            self.W2_raw[:] = -2.0
            self.a_raw[:]  = -1.0

    def encode(self, x):
        h = torch.relu(self.enc_hidden(x))
        h = self.enc_dropout(h)
        return torch.sigmoid(self.enc_out(h))

    @staticmethod
    def _constrain_item_param(logit, p_min, p_max):
        u = torch.sigmoid(logit)
        return p_min + (p_max - p_min) * u

    def _W2_pos_masked(self):
        W2_pos = F.softplus(self.W2_raw)  # >=0
        return W2_pos * self.Q_JK

    def decode(self, z_sample):
        eps = 1e-7
        a_k = torch.sigmoid(self.tau * z_sample).clamp(eps, 1.0 - eps)  # [N,K]

        W2 = self._W2_pos_masked().clamp(min=0.0)  # [J,K]
        wq = W2.unsqueeze(0)                       # [1,J,K]

        # AND pooling
        log_a = torch.log(a_k).unsqueeze(1)        # [N,1,K]
        log_m_and = (wq * log_a).sum(dim=2)        # [N,J]
        m_and = torch.exp(log_m_and).clamp(eps, 1.0 - eps)

        # OR pooling
        log_1ma = torch.log(1.0 - a_k).unsqueeze(1)
        log_prod_1ma = (wq * log_1ma).sum(dim=2)
        prod_1ma = torch.exp(log_prod_1ma).clamp(eps, 1.0 - eps)
        m_or = (1.0 - prod_1ma).clamp(eps, 1.0 - eps)

        # mixture gate
        pi = torch.sigmoid(self.gate_logit).unsqueeze(0)  # [1,J]
        base = (pi * m_and + (1.0 - pi) * m_or).clamp(eps, 1.0 - eps)

        # discrimination + intercept in logit space
        base_logit = torch.log(base) - torch.log(1.0 - base)
        a_j = F.softplus(self.a_raw).unsqueeze(0) + 1e-6
        b_j = self.b_j.unsqueeze(0)
        p0 = torch.sigmoid(a_j * base_logit + b_j).clamp(eps, 1.0 - eps)

        # slip/guess
        g = self._constrain_item_param(self.guess_logit, self.g_min, self.g_max).unsqueeze(0)
        s = self._constrain_item_param(self.slip_logit,  self.s_min, self.s_max).unsqueeze(0)
        p = (1.0 - s) * p0 + g * (1.0 - p0)
        return p.clamp(eps, 1.0 - eps)

    def forward(self, x, temperature=0.67):
        q = self.encode(x).clamp(1e-7, 1.0 - 1e-7)
        z = sample_concrete_bernoulli(q, temperature=float(temperature))
        recon = self.decode(z)
        return recon, z, q

@torch.no_grad()
def val_bce_thresholded(model, X_val, thr=0.5, device="cpu"):
    """
    X-only metric: encode -> hard threshold -> decode -> BCE(recon, X).
    This matches your simulation tuning metric.
    """
    model.eval()
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)

    q = model.encode(Xv).clamp(1e-7, 1.0 - 1e-7)     # [N,K]
    z_hard = (q > float(thr)).float()                # [N,K]
    recon = model.decode(z_hard)                     # [N,J]

    return float(F.binary_cross_entropy(recon, Xv).item())


@torch.no_grad()
def select_thr_xonly(model, X_tune, device="cpu",
                     grid=(0.2,0.3,0.4,0.5,0.6,0.7,0.8)):
    """
    Choose threshold that minimizes X-only thresholded reconstruction BCE.
    """
    best_thr, best_score = 0.5, float("inf")
    for thr in grid:
        score = val_bce_thresholded(model, X_tune, thr=thr, device=device)
        if score < best_score:
            best_score, best_thr = score, thr
    return float(best_thr), float(best_score)

@torch.no_grad()
def masked_val_bce(model, X_val, mask_val, temperature_eval=0.3, device="cpu"):
    model.eval()
    Xk = build_known_matrix_python(X_val, mask_val, fill_value=0.5)
    Xk_t = torch.tensor(Xk, dtype=torch.float32).to(device)
    pred = model(Xk_t, temperature=float(temperature_eval))[0].detach().cpu().numpy()
    y_true = X_val[mask_val].astype(float)
    y_prob = pred[mask_val]
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    bce = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()
    return float(bce)



def train_dvae_xonly_earlystop(
    model,
    X_train,
    X_val,
    mask_val,
    epochs=1200,
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=64,
    kl_weight=0.003,
    warmup_epochs=400,
    free_bits=0.02,
    temp_start=1.0,
    temp_end=0.3,
    mc_samples=3,
    gate_reg_weight=1e-3,
    w2_l1_weight=1e-4,
    sg_l2_weight=1e-4,      
    sg_init_p=0.15,         
    device="cpu",
    patience=20,
    eval_every=10,
):
    model.to(device)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xt),
        batch_size=min(int(batch_size), len(X_train)),
        shuffle=True,
        drop_last=False
    )
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_bce = float("inf")
    wait = 0

    for ep in range(int(epochs)):
        model.train()
        kl_scale = min(1.0, ep / max(1, int(warmup_epochs)))
        t = ep / max(1, epochs - 1)
        temperature = float(temp_start + (temp_end - temp_start) * t)

        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)

            recon_loss = 0.0
            kl_loss = 0.0
            for _ in range(int(mc_samples)):
                recon, _, q = model(xb, temperature=temperature)
                recon_loss += F.binary_cross_entropy(recon, xb)
                kl_loss    += kl_bernoulli_freebits(q, prior_p=0.5, free_bits=float(free_bits))
            recon_loss /= float(mc_samples)
            kl_loss    /= float(mc_samples)

            pi = torch.sigmoid(model.gate_logit)
            gate_reg = (pi * (1.0 - pi)).mean()

            W2_masked = model._W2_pos_masked()
            w2_l1 = W2_masked.abs().mean()

            loss = recon_loss + (kl_scale * float(kl_weight)) * kl_loss \
                   + float(gate_reg_weight) * gate_reg \
                   + float(w2_l1_weight) * w2_l1
            
            init_logit = torch.logit(torch.tensor(float(sg_init_p), device=device))
            sg_l2 = ((model.guess_logit - init_logit)**2).mean() + ((model.slip_logit - init_logit)**2).mean()
            loss = loss + float(sg_l2_weight) * sg_l2


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


        if (ep % int(eval_every) == 0) or (ep == epochs - 1):
            bce = masked_val_bce(
                model, X_val, mask_val,
                temperature_eval=float(temp_end),
                device=device
            )
            if bce < best_bce - 1e-6:
                best_bce = bce
                wait = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= int(patience) and best_state is not None:
                    model.load_state_dict(best_state)
                    break


    return model

def tune_dvae_simstyle_maskedval(Q, X_train, X_val, seed=SEED, n_candidates=40, device="cpu"):
    rng = np.random.RandomState(seed)
    mask_val = make_masks(X_val, prop=0.30, seed=seed)

    J, K = Q.shape

    search_space = {
        "hidden_units": [32, 64, 128, 256],
        "dropout": [0.0],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "kl_weight": [0.001, 0.003, 0.01],
        "free_bits": [0.0, 0.01, 0.02],
        "temp_start": [1.0],
        "temp_end": [0.2, 0.3, 0.5],
        "w2_l1_weight": [0.0, 1e-5, 1e-4, 3e-4],
        "gate_reg_weight": [1e-3],
        "epochs": [600, 1200],
    }

    candidates = list(ParameterSampler(search_space, n_iter=int(n_candidates), random_state=rng))

    best_params = None
    best_bce = float("inf")

    for ci, p in enumerate(candidates):
        set_global_seeds(seed + 1000 * ci)

        model = DVAECDM(
            num_items=J, num_attrs=K, Q=Q,
            hidden_units=int(p["hidden_units"]),
            dropout_rate=float(p["dropout"]),
        )

        model = train_dvae_xonly_earlystop(
            model,
            X_train=X_train,
            X_val=X_val,
            mask_val=mask_val,
            epochs=int(p["epochs"]),
            lr=float(p["lr"]),
            weight_decay=float(p["weight_decay"]),
            batch_size=64,
            kl_weight=float(p["kl_weight"]),
            warmup_epochs=400,
            free_bits=float(p["free_bits"]),
            temp_start=float(p["temp_start"]),
            temp_end=float(p["temp_end"]),
            mc_samples=3,
            gate_reg_weight=float(p["gate_reg_weight"]),
            w2_l1_weight=float(p["w2_l1_weight"]),
            sg_l2_weight=1e-4,
            sg_init_p=0.15,
            device=device,
            patience=20,
            eval_every=10
        )

        bce = masked_val_bce(model, X_val, mask_val, temperature_eval=float(p["temp_end"]), device=device)
        if bce < best_bce - 1e-6:
            best_bce = bce
            best_params = dict(p)
            best_params["val_masked_bce"] = float(best_bce)

    return best_params


# 5) Classical models via CDM + NPC

def to_r_matrix(X_np, mode='int'):
    if mode == 'int':
        vec = ro.IntVector(np.asarray(X_np).flatten(order='F'))
    else:
        vec = ro.FloatVector(np.asarray(X_np).flatten(order='F'))
    return ro.r.matrix(vec, nrow=np.asarray(X_np).shape[0], ncol=np.asarray(X_np).shape[1])

def fit_cdm_dina(X_train, Q):
    X_r = to_r_matrix(X_train, 'int')
    Q_r = to_r_matrix(Q, 'int')
    ro.r(f"set.seed({int(SEED)})")
    return CDM.din(X_r, q_matrix=Q_r, rule="DINA")

def fit_cdm_dino(X_train, Q):
    X_r = to_r_matrix(X_train, 'int')
    Q_r = to_r_matrix(Q, 'int')
    ro.r(f"set.seed({int(SEED)})")
    return CDM.din(X_r, q_matrix=Q_r, rule="DINO")

def fit_cdm_gdina(X_train, Q):
    X_r = to_r_matrix(X_train, 'int')
    Q_r = to_r_matrix(Q, 'int')
    ro.r(f"set.seed({int(SEED)})")
    return CDM.gdina(X_r, q_matrix=Q_r)

def predict_prob_cdm(fit, X_known):
    X_known_r = to_r_matrix(X_known, 'float')
    pred = ro.r('CDM::IRT.predict')(fit, X_known_r)
    expected = pred.rx2("expected")
    P = np.array(expected, dtype=float)
    if P.ndim == 3 and P.shape[1] == 1:
        P = np.squeeze(P, axis=1)
    elif P.ndim == 3 and P.shape[1] > 1:
        P = P.mean(axis=1)
    return P

def choose_gate_by_masked_bce(P_and, P_or, X_val, mask_val):
    y_true = X_val[mask_val].astype(float)

    eps = 1e-7
    p_and = np.clip(P_and[mask_val], eps, 1 - eps)
    p_or  = np.clip(P_or [mask_val], eps, 1 - eps)

    bce_and = -(y_true * np.log(p_and) + (1 - y_true) * np.log(1 - p_and)).mean()
    bce_or  = -(y_true * np.log(p_or ) + (1 - y_true) * np.log(1 - p_or )).mean()

    return ("AND" if bce_and <= bce_or else "OR"), float(bce_and), float(bce_or)


def npc_alpha_then_prob_with_mask(X_known_r, Q, mask, gate_type="AND", method="Weighted", wg=1.0, ws=1.0):
    N, J = X_known_r.shape
    K = Q.shape[1]
    P = np.zeros((N, J), dtype=float)

    for i in range(N):
        obs_idx = np.where(~mask[i])[0]
        if obs_idx.size == 0:
            P[i, :] = 0.5
            continue

        y_i = X_known_r[i, obs_idx]
        y_i = np.nan_to_num(y_i, nan=0.0).astype(int)

        Y_r = to_r_matrix(y_i.reshape(1, -1), 'int')
        Q_sub = Q[obs_idx, :]
        Q_r = to_r_matrix(Q_sub, 'int')

        ro.r(f"set.seed({int(SEED)})")
        res = NPCD.AlphaNP(Y=Y_r, Q=Q_r, gate=str(gate_type), method=method, wg=float(wg), ws=float(ws))
        alpha_i = np.array(res.rx2("alpha.est")).astype(int).reshape(1, K)

        for j in range(J):
            req = Q[j].astype(bool)
            if not req.any():
                P[i, j] = 0.5
            else:
                if str(gate_type).upper() == "AND":
                    P[i, j] = float(alpha_i[0, req].sum() == req.sum())
                else:
                    P[i, j] = float(alpha_i[0, req].sum() >= 1)
    return P


# 6) Main dataset runner

def run_dataset(data_path, q_path, results_dir, dataset_name, unseen_prop=0.30, seed=SEED, n_candidates=40):
    os.makedirs(results_dir, exist_ok=True)
    set_global_seeds(seed)

    X, Q = load_from_csv(data_path, q_path)
    N, J = X.shape
    K = Q.shape[1]
    print(f"\n[{dataset_name}] seed={seed} | N={N}, J={J}, K={K}")

    idx_tr, idx_val, idx_te = split_students(X, 0.7, 0.15, 0.15, seed=seed)
    X_tr, X_val, X_te = X[idx_tr], X[idx_val], X[idx_te]

    # ---- Tune DVAE 
    best = tune_dvae_simstyle_maskedval(Q, X_tr, X_val, seed=seed, n_candidates=n_candidates, device=device)
    print(f"[{dataset_name}] Best DVAE params:", best)

    # ---- Retrain DVAE on train+val (no early stop)
    X_trv = np.vstack([X_tr, X_val])
    model = DVAECDM(
        num_items=J, num_attrs=K, Q=Q,
        hidden_units=int(best["hidden_units"]),
        dropout_rate=float(best["dropout"]),
    ).to(device)

    # full training 
    dummy_mask = make_masks(X_val, prop=unseen_prop, seed=seed)  
    model = train_dvae_xonly_earlystop(
        model,
        X_train=X_trv,
        X_val=X_val,               
        mask_val=dummy_mask,
        epochs=int(best["epochs"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        batch_size=64,
        kl_weight=float(best["kl_weight"]),
        warmup_epochs=400,
        free_bits=float(best["free_bits"]),
        temp_start=float(best["temp_start"]),
        temp_end=float(best["temp_end"]),
        mc_samples=3,
        gate_reg_weight=float(best["gate_reg_weight"]),
        w2_l1_weight=float(best["w2_l1_weight"]),
        sg_l2_weight=1e-4,
        sg_init_p=0.15,
        device=device,
        patience=10**9,             
        eval_every=10**9
    )
    model.eval()

    # ---- Test masking (same for all models)
    test_mask = make_masks(X_te, prop=unseen_prop, seed=seed)
    X_te_known_py = build_known_matrix_python(X_te, test_mask, fill_value=0.5)
    X_te_known_r  = build_known_matrix_r(X_te, test_mask)

    # ---- DVAE predictions
    with torch.no_grad():
        P_dvae = model(
            torch.tensor(X_te_known_py, dtype=torch.float32).to(device),
            temperature=float(best["temp_end"])
        )[0].detach().cpu().numpy()
    met_dvae = evaluate_on_unseen(X_te[test_mask], P_dvae[test_mask], thr=0.5)

    # ---- Classical CDMs 
    fit_dina  = fit_cdm_dina(X_tr, Q)
    fit_dino  = fit_cdm_dino(X_tr, Q)
    fit_gdina = fit_cdm_gdina(X_tr, Q)

    P_dina  = predict_prob_cdm(fit_dina,  X_te_known_r)
    P_dino  = predict_prob_cdm(fit_dino,  X_te_known_r)
    P_gdina = predict_prob_cdm(fit_gdina, X_te_known_r)

    met_dina  = evaluate_on_unseen(X_te[test_mask], P_dina [test_mask], thr=0.5)
    met_dino  = evaluate_on_unseen(X_te[test_mask], P_dino [test_mask], thr=0.5)
    met_gdina = evaluate_on_unseen(X_te[test_mask], P_gdina[test_mask], thr=0.5)

    # ---- NPC (per-student using only observed items)
    
    P_npc_and = npc_alpha_then_prob_with_mask(
        X_te_known_r, Q, test_mask, gate_type="AND", method="Hamming", wg=1.0, ws=1.0
    )
    P_npc_or  = npc_alpha_then_prob_with_mask(
        X_te_known_r, Q, test_mask, gate_type="OR",  method="Hamming", wg=1.0, ws=1.0
    )

    met_npc_and = evaluate_on_unseen(X_te[test_mask], P_npc_and[test_mask], thr=0.5)
    met_npc_or  = evaluate_on_unseen(X_te[test_mask], P_npc_or [test_mask], thr=0.5)

    # ---- Save
    def row(name, m):
        return {"dataset": dataset_name, "seed": int(seed), "model": name,
                "ACC": m["ACC"], "AUC": m["AUC"], "BCE": m["BCE"], "ERROR": m["ERROR"]}

    df = pd.DataFrame([
        row("DVAE",     met_dvae),
        row("DINA",     met_dina),
        row("DINO",     met_dino),
        row("GDINA",    met_gdina),
        row("NPC_AND",  met_npc_and),
        row("NPC_OR",   met_npc_or),
    ])

    out_csv = os.path.join(results_dir, f"{dataset_name}_mask{int(unseen_prop*100)}_seed{int(seed)}.csv")
    df.to_csv(out_csv, index=False)

    
    params_csv = os.path.join(results_dir, f"{dataset_name}_DVAE_bestparams_seed{int(seed)}.csv")
    pd.DataFrame([best]).to_csv(params_csv, index=False)

    print(f"[{dataset_name}] Results\n{df.to_string(index=False)}")
    print(f"[{dataset_name}] Saved: {out_csv}")
    print(f"[{dataset_name}] DVAE params saved: {params_csv}")
    return df, best


# 7) Run both datasets across multiple seeds

if __name__ == "__main__":
    RESULTS_DIR = "results_realdata_simstyle_dvae"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    SEEDS = [11, 22, 33, 44, 55]
    all_results = []

    for s in SEEDS:
        df_ecpe, _ = run_dataset(
            "ecpe_data.csv", "ecpe_q.csv",
            results_dir=RESULTS_DIR, dataset_name="ECPE",
            unseen_prop=0.10, seed=s, n_candidates=5
        )
        df_frac, _ = run_dataset(
            "fraction_data.csv", "fraction_q.csv",
            results_dir=RESULTS_DIR, dataset_name="FRACTION",
            unseen_prop=0.10, seed=s, n_candidates=5
        )
        all_results.append(pd.concat([df_ecpe, df_frac], ignore_index=True))

    final = pd.concat(all_results, ignore_index=True)

    # Mean/SD across seeds
    summary = (
        final.groupby(["dataset", "model"], as_index=False)[["ACC", "AUC", "BCE", "ERROR"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_csv = os.path.join(RESULTS_DIR, "summary_across_seeds.csv")
    summary.to_csv(summary_csv, index=False)

    print("\n=== Summary across seeds ===")
    print(summary)
    print(f"Saved summary: {summary_csv}")
