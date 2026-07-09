import os as os
os.getcwd()
os.chdir('')
os.getcwd()


import time
import random
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold, ParameterSampler

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()

GDINA = importr("GDINA")
NPCD  = importr("NPCD")
CDM   = importr("CDM")



# Reproducibility

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"



# 1) Simulation parameters

Q_MATRICES = {
    "Q1": {  # 3 attributes
        "matrix": np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]),
        "num_attrs": 3
    },
    "Q2": {  # 4 attributes
        "matrix": np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,1,0,1],
            [1,0,1,0],
            [1,1,1,0],
            [0,1,1,1],
            [0,1,1,0],
            [0,0,1,1],
            [1,0,0,1]
        ]),
        "num_attrs": 4
    },
    "Q3": {  # 8 attributes 
        "matrix": np.array([
            [0,0,0,1,0,1,1,0],
            [0,0,0,1,0,0,1,0],
            [0,0,0,1,0,0,1,0],
            [0,1,1,0,1,0,1,0],
            [0,1,0,1,0,0,1,1],
            [0,0,0,0,0,0,1,0],
            [1,1,0,0,0,0,1,0],
            [0,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,0,0],
            [0,1,0,0,1,0,1,1],
            [0,1,0,0,1,0,1,0],
            [0,0,0,0,0,0,1,1],
            [0,1,0,1,1,0,1,0],
            [0,1,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [0,1,0,0,0,0,1,0],
            [0,1,0,0,1,0,1,0],
            [0,1,0,0,1,1,1,0],
            [1,1,1,0,1,0,1,0],
            [0,1,1,0,1,0,1,0],
            [1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1]
        ]),
        "num_attrs": 8
    }
}



# 2) Synthetic data generation

def generate_attributes(N, K, p=0.5):
    return np.random.binomial(1, p, size=(N, K))

def generate_responses(attributes, Q, gate_type, slip=None, guess=None):
    """
    DINA/DINO-like generation: AND or OR or mixed gate
    slip/guess sampled per item if None.
    """
    N, K = attributes.shape
    J = len(Q)
    X = np.zeros((N, J), dtype=int)

    if slip is None:
        slip = np.random.uniform(0.05, 0.35, J)
    if guess is None:
        guess = np.random.uniform(0.05, 0.35, J)

    if gate_type == "mixed":
        mixed_gates = np.random.choice(["AND", "OR"], size=J)

    for j in range(J):
        req_attrs = np.where(np.array(Q[j]).astype(bool))[0]
        gate = gate_type if gate_type != "mixed" else mixed_gates[j]

        for i in range(N):
            if req_attrs.size == 0:
                mastered = True
            else:
                if gate == "AND":
                    mastered = np.all(attributes[i, req_attrs] == 1)
                else:
                    mastered = np.any(attributes[i, req_attrs] == 1)

            prob = mastered * (1 - slip[j]) + (1 - mastered) * guess[j]
            X[i, j] = np.random.binomial(1, prob)

    return X, slip, guess



# 3) DVAE utilities 


def sample_concrete_bernoulli(q, temperature=0.67, eps=1e-7):
    """
    Concrete / Gumbel-Sigmoid relaxation for Bernoulli:
      z = sigmoid((logit(q) + g) / temp)
    where g = log(u) - log(1-u), u~Uniform(0,1).
    """
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

    if free_bits is not None and free_bits > 0:
        kl_dim = torch.clamp(kl_dim, min=float(free_bits))

    return kl_dim.sum(dim=1).mean()

def smooth_binary_targets(x, eps=0.0):
    if eps <= 0:
        return x
    return x * (1.0 - 2.0 * eps) + eps






# 4) DVAE-CDM model 




class DVAECDM(nn.Module):
    """
    Encoder:  X -> q(Z=1|X)
    Sample:   z ~ Concrete(q, temp)   (replaces inverse-CDF)
    Decoder:  uses Q-masked nonnegative W2 + item discrimination a_j + bias b_j
             supports AND/OR mixture via per-item gate pi_j
             then applies explicit slip/guess.
    """
    def __init__(
        self,
        num_items,
        num_attrs,
        Q,
        hidden_units=64,
        dropout_rate=0.0,
        # slip/guess bounds
        g_min=1e-4, g_max=0.499,
        s_min=1e-4, s_max=0.499,
        # mastery sharpness inside AND/OR formulas
        tau=5.0,
        init_gate=0.5
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.num_attrs = int(num_attrs)

        Q_tensor = torch.tensor(np.asarray(Q), dtype=torch.float32)  # [J,K]
        self.register_buffer("Q_JK", Q_tensor)                       # [J,K]
        self.register_buffer("Q_KJ", Q_tensor.t().contiguous())       # [K,J]

        # ----- Encoder -----
        self.enc_hidden  = nn.Linear(self.num_items, hidden_units)
        self.enc_dropout = nn.Dropout(dropout_rate)
        self.enc_out     = nn.Linear(hidden_units, self.num_attrs)

        # ----- Decoder params -----
        # W2: item x attr, masked by Q, constrained nonnegative via softplus
        self.W2_raw = nn.Parameter(torch.zeros(self.num_items, self.num_attrs))

        # item discrimination a_j > 0 and item intercept b_j
        self.a_raw = nn.Parameter(torch.zeros(self.num_items))  # -> softplus
        self.b_j   = nn.Parameter(torch.zeros(self.num_items))

        # Slip/Guess (item-wise), constrained to [min,max]
        init_p = 0.15
        init_logit = torch.logit(torch.full((self.num_items,), float(init_p)))
        self.guess_logit = nn.Parameter(init_logit.clone())
        self.slip_logit  = nn.Parameter(init_logit.clone())
        self.g_min, self.g_max = float(g_min), float(g_max)
        self.s_min, self.s_max = float(s_min), float(s_max)

        # Per-item gate mixture pi_j = sigmoid(gate_logit_j)
        gate_init_logit = torch.logit(torch.full((self.num_items,), float(init_gate)))
        self.gate_logit = nn.Parameter(gate_init_logit.clone())

        self.tau = float(tau)

        # Init encoder
        nn.init.xavier_uniform_(self.enc_hidden.weight)
        nn.init.zeros_(self.enc_hidden.bias)
        nn.init.xavier_uniform_(self.enc_out.weight)
        nn.init.zeros_(self.enc_out.bias)

        # Init W2 to small positive on Q entries (helps convergence)
        with torch.no_grad():
            self.W2_raw[:] = -2.0  # softplus(-2) ~ 0.13
            self.a_raw[:] = -1.0   # softplus(-1) ~ 0.31

    def encode(self, x):
        h = torch.relu(self.enc_hidden(x))
        h = self.enc_dropout(h)
        return torch.sigmoid(self.enc_out(h))

    @staticmethod
    def _constrain_item_param(logit, p_min, p_max):
        u = torch.sigmoid(logit)
        return p_min + (p_max - p_min) * u

    def _W2_pos_masked(self):
        # nonnegative weights, only allow where Q==1
        W2_pos = F.softplus(self.W2_raw)  # [J,K] >= 0
        return W2_pos * self.Q_JK         # mask by Q

    def decode(self, z_sample):
        """
        z_sample: [N,K] continuous (Concrete sample)
        return p(X=1): [N,J]
        """
        eps = 1e-7

        # attribute "mastery probabilities" used in AND/OR pooling
        a_k = torch.sigmoid(self.tau * z_sample).clamp(eps, 1.0 - eps)  # [N,K]

        # Q-masked nonnegative W2
        W2 = self._W2_pos_masked().clamp(min=0.0)  # [J,K]

        # ---------- Weighted AND ----------
        # log m_and_j = sum_k (q_jk * w_jk * log(a_k))
        log_a = torch.log(a_k).unsqueeze(1)        # [N,1,K]
        wq = (W2).unsqueeze(0)                     # [1,J,K]
        log_m_and = (wq * log_a).sum(dim=2)        # [N,J]
        m_and = torch.exp(log_m_and).clamp(eps, 1.0 - eps)

        # ---------- Weighted OR ----------
        # m_or_j = 1 - prod_k (1-a_k)^(q_jk*w_jk)
        log_1ma = torch.log(1.0 - a_k).unsqueeze(1)  # [N,1,K]
        log_prod_1ma = (wq * log_1ma).sum(dim=2)     # [N,J]
        prod_1ma = torch.exp(log_prod_1ma).clamp(eps, 1.0 - eps)
        m_or = (1.0 - prod_1ma).clamp(eps, 1.0 - eps)

        # Mixture per item (learns AND/OR/MIXED without gate labels)
        pi = torch.sigmoid(self.gate_logit).unsqueeze(0)  # [1,J]
        base = (pi * m_and + (1.0 - pi) * m_or).clamp(eps, 1.0 - eps)

        # ---------- Discrimination + intercept in logit space ----------
        base_logit = torch.log(base) - torch.log(1.0 - base)  # [N,J]
        a_j = F.softplus(self.a_raw).unsqueeze(0) + 1e-6       # [1,J] > 0
        b_j = self.b_j.unsqueeze(0)                            # [1,J]
        p0 = torch.sigmoid(a_j * base_logit + b_j).clamp(eps, 1.0 - eps)

        # ---------- Slip/guess ----------
        g = self._constrain_item_param(self.guess_logit, self.g_min, self.g_max).unsqueeze(0)
        s = self._constrain_item_param(self.slip_logit,  self.s_min, self.s_max).unsqueeze(0)

        p = (1.0 - s) * p0 + g * (1.0 - p0)
        return p.clamp(eps, 1.0 - eps)

    def forward(self, x, temperature=0.67):
        q = self.encode(x).clamp(1e-7, 1.0 - 1e-7)  # [N,K]
        z = sample_concrete_bernoulli(q, temperature=float(temperature))  # [N,K]
        recon = self.decode(z)  # [N,J]
        return recon, z, q



# 5 Training (X-only) with gate regularization




def train_dvae_xonly(
    model,
    X_train,
    epochs=1200,
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=64,

    # KL annealing + free bits
    kl_weight=0.003,
    warmup_epochs=400,
    free_bits=0.02,

    # temperature schedule (better relaxation)
    temp_start=1.0,
    temp_end=0.3,

    # MC samples
    mc_samples=3,

    # regularization
    grad_clip=5.0,
    label_smoothing_eps=0.0,

    # gate + sparsity
    gate_reg_weight=1e-3,   # pushes pi to 0 or 1
    w2_l1_weight=1e-4,      # sparsity on W2 within Q

    device="cpu",
    verbose=False
):
    """
    Loss = BCE(recon, X)
         + (kl_scale*kl_weight)*KL_freebits(q||prior)
         + gate_reg_weight * mean(pi*(1-pi))
         + w2_l1_weight * mean(|W2_pos * Q|)
    """
    model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xt),
        batch_size=min(int(batch_size), len(X_train)),
        shuffle=True,
        drop_last=False
    )

    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    for ep in range(int(epochs)):
        model.train()

        # KL warmup
        kl_scale = min(1.0, ep / max(1, int(warmup_epochs)))

        # temperature schedule
        t = ep / max(1, epochs - 1)
        temperature = float(temp_start + (temp_end - temp_start) * t)

        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)

            xb_tgt = smooth_binary_targets(xb, eps=float(label_smoothing_eps))

            opt.zero_grad(set_to_none=True)

            recon_loss = 0.0
            kl_loss = 0.0

            for _ in range(int(mc_samples)):
                recon, _, q = model(xb, temperature=temperature)
                recon_loss = recon_loss + F.binary_cross_entropy(recon, xb_tgt)
                kl_loss    = kl_loss    + kl_bernoulli_freebits(q, prior_p=0.5, free_bits=float(free_bits))

            recon_loss = recon_loss / float(mc_samples)
            kl_loss    = kl_loss    / float(mc_samples)

            # Gate regularizer: minimize pi(1-pi) -> pushes pi to 0 or 1
            pi = torch.sigmoid(model.gate_logit)
            gate_reg = (pi * (1.0 - pi)).mean()

            # Sparsity on W2 within Q
            W2_masked = model._W2_pos_masked()
            w2_l1 = W2_masked.abs().mean()

            loss = (
                recon_loss
                + (kl_scale * float(kl_weight)) * kl_loss
                + float(gate_reg_weight) * gate_reg
                + float(w2_l1_weight) * w2_l1
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            ep_loss += float(loss.detach().cpu().item())

        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(
                f"[ep {ep:4d}] loss={ep_loss/max(1,len(loader)):.4f} "
                f"temp={temperature:.3f} kl_scale={kl_scale:.3f} "
                f"pi_mean={torch.sigmoid(model.gate_logit).mean().item():.3f} "
                f"W2_l1={w2_l1.item():.4f}"
            )

    return model


@torch.no_grad()
def eval_val_bce_mc(
    model,
    X_val,
    temperature_eval=0.3,
    mc_eval=10,
    label_smoothing_eps=0.0,
    device="cpu"
):
    model.eval()
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    Xv_tgt = smooth_binary_targets(Xv, eps=float(label_smoothing_eps))

    bces = []
    for _ in range(int(mc_eval)):
        recon, _, _ = model(Xv, temperature=float(temperature_eval))
        bces.append(F.binary_cross_entropy(recon, Xv_tgt).item())
    return float(np.mean(bces))




# 6) X-only tuning



HIDDEN_UNITS_CANDIDATES = [32, 64, 128, 256]
DROPOUT_CANDIDATES = [0.0]
LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3]
WEIGHT_DECAY_CANDIDATES = [0.0, 1e-6, 1e-5, 1e-4]
KL_WEIGHT_CANDIDATES = [0.001, 0.003, 0.01]


TEMP_START_CANDIDATES = [1.0]
TEMP_END_CANDIDATES = [0.2, 0.3, 0.5]


FREE_BITS_CANDIDATES = [0.0, 0.01, 0.02]
W2_L1_CANDIDATES = [0.0, 1e-5, 1e-4, 3e-4]


GATE_REG_CANDIDATES = [1e-3]


def tune_hyperparameters_xonly(
    q_info,
    X,
    n_candidates=40,
    n_splits=5,
    n_restarts=3,
    tuning_epochs=1200,
    warmup_epochs=400,
    batch_size=64,
    mc_samples=3,
    mc_eval=10,
    label_smoothing_eps=0.0,   
    random_state=43,
    device="cpu",
):

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))

    search_space = {
        "hidden_units": HIDDEN_UNITS_CANDIDATES,
        "dropout": DROPOUT_CANDIDATES,
        "lr": LR_CANDIDATES,
        "temp_start": TEMP_START_CANDIDATES,
        "temp_end": TEMP_END_CANDIDATES,
        "kl_weight": KL_WEIGHT_CANDIDATES,
        "weight_decay": WEIGHT_DECAY_CANDIDATES,
        "free_bits": FREE_BITS_CANDIDATES,
        "w2_l1_weight": W2_L1_CANDIDATES,
        "gate_reg_weight": GATE_REG_CANDIDATES,
    }

    sampler = list(
        ParameterSampler(search_space, n_iter=int(n_candidates), random_state=int(random_state))
    )
    folds = list(kf.split(X))

    best_cv_bce = float("inf")
    best_params = None

    for p in sampler:
        fold_best_bces = []

        for train_idx, val_idx in folds:
            X_train, X_val = X[train_idx], X[val_idx]
            best_restart_bce = float("inf")

            for r in range(int(n_restarts)):
                torch.manual_seed(SEED + 1000 * r + 7)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + 1000 * r + 7)

                model = DVAECDM(
                    num_items=len(q_info["matrix"]),
                    num_attrs=q_info["num_attrs"],
                    Q=q_info["matrix"],
                    hidden_units=int(p["hidden_units"]),
                    dropout_rate=float(p["dropout"]),
                )

                train_dvae_xonly(
                    model,
                    X_train=X_train,
                    epochs=int(tuning_epochs),
                    lr=float(p["lr"]),
                    weight_decay=float(p["weight_decay"]),
                    batch_size=int(min(batch_size, len(X_train))),

                    # KL annealing + free bits
                    kl_weight=float(p["kl_weight"]),
                    warmup_epochs=int(warmup_epochs),
                    free_bits=float(p["free_bits"]),

                    # temperature schedule
                    temp_start=float(p["temp_start"]),
                    temp_end=float(p["temp_end"]),

                    # MC
                    mc_samples=int(mc_samples),

                    # target smoothing (usually 0.0 for CDM)
                    label_smoothing_eps=float(label_smoothing_eps),

                    # gate + sparsity
                    gate_reg_weight=float(p["gate_reg_weight"]),
                    w2_l1_weight=float(p["w2_l1_weight"]),

                    device=device,
                    verbose=False
                )

                val_bce = eval_val_bce_mc(
                    model,
                    X_val=X_val,
                    temperature_eval=float(p["temp_end"]),
                    mc_eval=int(mc_eval),
                    label_smoothing_eps=float(label_smoothing_eps),
                    device=device
                )

                if val_bce < best_restart_bce:
                    best_restart_bce = val_bce

            fold_best_bces.append(best_restart_bce)

        cv_bce = float(np.mean(fold_best_bces))

        if cv_bce < best_cv_bce:
            best_cv_bce = cv_bce
            best_params = dict(p)
            best_params["val_bce_cv"] = best_cv_bce

    return best_params







# 7) Traditional CDM baselines

def process_q_for_traditional(Q):
    Q_clean = []
    for item in Q:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item[0], list):
            combined = np.any(item, axis=0).astype(int)
            Q_clean.append(combined.tolist())
        else:
            Q_clean.append(item)
    return np.array(Q_clean)

def run_traditional_cdm(X, Q, model_type):
    Q_processed = process_q_for_traditional(Q)
    X_r = ro.r.matrix(ro.IntVector(X.flatten(order="F")), nrow=X.shape[0], ncol=X.shape[1])
    Q_r = ro.r.matrix(ro.IntVector(Q_processed.flatten(order="F")), nrow=Q_processed.shape[0], ncol=Q_processed.shape[1])
    fit = GDINA.GDINA(X_r, Q=Q_r, model=model_type, verbose=0)
    return np.array(ro.r["personparm"](fit), dtype=int)



def run_npc(X, Q, gate_type="AND", method="Weighted", wg=1, ws=1):
    
    Q_processed = process_q_for_traditional(Q)
    

    X_r = ro.r.matrix(ro.IntVector(X.flatten(order='F')), 
                     nrow=X.shape[0], ncol=X.shape[1])
    Q_r = ro.r.matrix(ro.IntVector(Q_processed.flatten(order='F')), 
                     nrow=Q_processed.shape[0], ncol=Q_processed.shape[1])
    

    if gate_type == 'mixed':
        gate_npc = str(np.random.choice(['AND', 'OR']))
    else:
        gate_npc = str(gate_type)
    npc_result = NPCD.AlphaNP(Y=X_r, Q=Q_r, gate=gate_npc, method=method, wg=float(wg), ws=float(ws))
    
    
 
    alpha_est = np.array(npc_result.rx2("alpha.est"))
    return alpha_est.astype(int)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


            
def main():
    RESULTS_DIR = "results_sim_dvae_new3"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    CONDITIONS = [
        {'q_matrix': 'Q1', 'gate_type': 'AND', 'sample_size': 30},
        {'q_matrix': 'Q1', 'gate_type': 'AND', 'sample_size': 50},
        {'q_matrix': 'Q1', 'gate_type': 'AND', 'sample_size': 200},
        {'q_matrix': 'Q1', 'gate_type': 'AND', 'sample_size': 500},
        {'q_matrix': 'Q1', 'gate_type': 'AND', 'sample_size': 1000},
        {'q_matrix': 'Q2', 'gate_type': 'AND', 'sample_size': 30},
        {'q_matrix': 'Q2', 'gate_type': 'AND', 'sample_size': 50},
        {'q_matrix': 'Q2', 'gate_type': 'AND', 'sample_size': 200},
        {'q_matrix': 'Q2', 'gate_type': 'AND', 'sample_size': 500},
        {'q_matrix': 'Q2', 'gate_type': 'AND', 'sample_size': 1000},
        {'q_matrix': 'Q3', 'gate_type': 'AND', 'sample_size': 30},
        {'q_matrix': 'Q3', 'gate_type': 'AND', 'sample_size': 50},
        {'q_matrix': 'Q3', 'gate_type': 'AND', 'sample_size': 200},
        {'q_matrix': 'Q3', 'gate_type': 'AND', 'sample_size': 500},
        {'q_matrix': 'Q3', 'gate_type': 'AND', 'sample_size': 1000},
        
        {'q_matrix': 'Q1', 'gate_type': 'OR', 'sample_size': 30},
        {'q_matrix': 'Q1', 'gate_type': 'OR', 'sample_size': 50},
        {'q_matrix': 'Q1', 'gate_type': 'OR', 'sample_size': 200},
        {'q_matrix': 'Q1', 'gate_type': 'OR', 'sample_size': 500},
        {'q_matrix': 'Q1', 'gate_type': 'OR', 'sample_size': 1000},
        {'q_matrix': 'Q2', 'gate_type': 'OR', 'sample_size': 30},
        {'q_matrix': 'Q2', 'gate_type': 'OR', 'sample_size': 50},
        {'q_matrix': 'Q2', 'gate_type': 'OR', 'sample_size': 200},
        {'q_matrix': 'Q2', 'gate_type': 'OR', 'sample_size': 500},
        {'q_matrix': 'Q2', 'gate_type': 'OR', 'sample_size': 1000},
        {'q_matrix': 'Q3', 'gate_type': 'OR', 'sample_size': 30},
        {'q_matrix': 'Q3', 'gate_type': 'OR', 'sample_size': 50},
        {'q_matrix': 'Q3', 'gate_type': 'OR', 'sample_size': 200},
        {'q_matrix': 'Q3', 'gate_type': 'OR', 'sample_size': 500},
        {'q_matrix': 'Q3', 'gate_type': 'OR', 'sample_size': 1000},
        
        {'q_matrix': 'Q1', 'gate_type': 'mixed', 'sample_size': 30},
        {'q_matrix': 'Q1', 'gate_type': 'mixed', 'sample_size': 50},
        {'q_matrix': 'Q1', 'gate_type': 'mixed', 'sample_size': 200},
        {'q_matrix': 'Q1', 'gate_type': 'mixed', 'sample_size': 500},
        {'q_matrix': 'Q1', 'gate_type': 'mixed', 'sample_size': 1000},
        {'q_matrix': 'Q2', 'gate_type': 'mixed', 'sample_size': 30},
        {'q_matrix': 'Q2', 'gate_type': 'mixed', 'sample_size': 50},
        {'q_matrix': 'Q2', 'gate_type': 'mixed', 'sample_size': 200},
        {'q_matrix': 'Q2', 'gate_type': 'mixed', 'sample_size': 500},
        {'q_matrix': 'Q2', 'gate_type': 'mixed', 'sample_size': 1000},
        {'q_matrix': 'Q3', 'gate_type': 'mixed', 'sample_size': 30},
        {'q_matrix': 'Q3', 'gate_type': 'mixed', 'sample_size': 50},
        {'q_matrix': 'Q3', 'gate_type': 'mixed', 'sample_size': 200},
        {'q_matrix': 'Q3', 'gate_type': 'mixed', 'sample_size': 500},
        {'q_matrix': 'Q3', 'gate_type': 'mixed', 'sample_size': 1000},
        
        
        
        
    ]

    REPLICATIONS = 100  

    results = []

    for c_idx, condition in enumerate(CONDITIONS):
        print(f"\n=== Processing Condition: {condition} ===")
        q_info = Q_MATRICES[condition["q_matrix"]]
        cond_results = []

        
        # (A) Tune ONCE per condition (using a fixed tuning dataset)
        
        tune_seed = SEED + 100000 * c_idx + 999
        seed_everything(tune_seed)

        true_attrs_tune = generate_attributes(condition["sample_size"], q_info["num_attrs"])
        X_tune, _, _ = generate_responses(true_attrs_tune, q_info["matrix"], condition["gate_type"])

        print("Tuning DVAE hyperparameters ONCE for this condition (X-only)...")
        best_params = tune_hyperparameters_xonly(
            q_info=q_info,
            X=X_tune,
            n_candidates=3,
            n_splits=5,
            n_restarts=1,
            tuning_epochs=600,
            warmup_epochs=400,
            batch_size=64,
            mc_samples=3,
            mc_eval=10,
            label_smoothing_eps=0.0,
            random_state=tune_seed,
            device=device
        )
        print(f"[Condition-level best params] {best_params}")

        
        # (B) Replications: NO tuning here, reuse best_params
        
        for rep in range(REPLICATIONS):
            rep_seed = SEED + 100000 * c_idx + rep
            seed_everything(rep_seed)

            print(f"\n--- Replication {rep+1}/{REPLICATIONS} ---")

            true_attrs = generate_attributes(condition["sample_size"], q_info["num_attrs"])
            X, slip, guess = generate_responses(true_attrs, q_info["matrix"], condition["gate_type"])
            avg_slip = float(np.mean(slip))
            avg_guess = float(np.mean(guess))

            # ----- Final DVAE training on this replication's X (fixed params)
            start_time = time.time()
            model = DVAECDM(
                num_items=len(q_info["matrix"]),
                num_attrs=q_info["num_attrs"],
                Q=q_info["matrix"],
                hidden_units=int(best_params["hidden_units"]),
                dropout_rate=float(best_params["dropout"]),
            )

            train_dvae_xonly(
                model,
                X_train=X,
                epochs=1200,
                lr=float(best_params["lr"]),
                weight_decay=float(best_params["weight_decay"]),
                batch_size=64,
                kl_weight=float(best_params["kl_weight"]),
                warmup_epochs=400,
                free_bits=float(best_params["free_bits"]),
                temp_start=float(best_params["temp_start"]),
                temp_end=float(best_params["temp_end"]),
                mc_samples=3,
                label_smoothing_eps=0.0,
                gate_reg_weight=float(best_params["gate_reg_weight"]),
                w2_l1_weight=float(best_params["w2_l1_weight"]),
                device=device,
                verbose=False
            )
            dvae_time = time.time() - start_time

            # ----- DVAE evaluation (true_attrs used ONLY here)
            thr = 0.5
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                q_prob = model.encode(X_tensor).cpu().numpy()
            dvae_est = (q_prob > thr).astype(int)

            dvae_error = float(np.mean(true_attrs != dvae_est))
            dvae_accuracy = float(np.mean(true_attrs == dvae_est))
            dvae_PAR = float(np.mean(np.all(true_attrs == dvae_est, axis=1)))

            # ----- Baselines 
            Q_processed = process_q_for_traditional(q_info["matrix"])

            start_time = time.time()
            dina_est = run_traditional_cdm(X, Q_processed, "DINA")
            dina_time = time.time() - start_time
            dina_error = float(np.mean(true_attrs != dina_est))
            dina_accuracy = float(np.mean(true_attrs == dina_est))
            dina_PAR = float(np.mean(np.all(true_attrs == dina_est, axis=1)))

            start_time = time.time()
            dino_est = run_traditional_cdm(X, Q_processed, "DINO")
            dino_time = time.time() - start_time
            dino_error = float(np.mean(true_attrs != dino_est))
            dino_accuracy = float(np.mean(true_attrs == dino_est))
            dino_PAR = float(np.mean(np.all(true_attrs == dino_est, axis=1)))

            start_time = time.time()
            gdina_est = run_traditional_cdm(X, Q_processed, "GDINA")
            gdina_time = time.time() - start_time
            gdina_error = float(np.mean(true_attrs != gdina_est))
            gdina_accuracy = float(np.mean(true_attrs == gdina_est))
            gdina_PAR = float(np.mean(np.all(true_attrs == gdina_est, axis=1)))

            start_time = time.time()
            npc_est = run_npc(
                X, Q_processed,
                gate_type=condition["gate_type"],
                method="Weighted",
                wg=avg_guess, ws=avg_slip
            )
            npc_time = time.time() - start_time
            npc_error = float(np.mean(true_attrs != npc_est))
            npc_accuracy = float(np.mean(true_attrs == npc_est))
            npc_PAR = float(np.mean(np.all(true_attrs == npc_est, axis=1)))

            cond_results.append({
                "DVAE": {"error": dvae_error, "accuracy": dvae_accuracy, "PAR": dvae_PAR, "time": dvae_time},
                "DINA": {"error": dina_error, "accuracy": dina_accuracy, "PAR": dina_PAR, "time": dina_time},
                "DINO": {"error": dino_error, "accuracy": dino_accuracy, "PAR": dino_PAR, "time": dino_time},
                "GDINA": {"error": gdina_error, "accuracy": gdina_accuracy, "PAR": gdina_PAR, "time": gdina_time},
                "NPC": {"error": npc_error, "accuracy": npc_accuracy, "PAR": npc_PAR, "time": npc_time},
                "params": best_params
            })




             
            row = {
                "condition_q": condition["q_matrix"],
                "gate_type": condition["gate_type"],
                "sample_size": condition["sample_size"],
                "replication": rep + 1,

                "DVAE_error": dvae_error, "DVAE_accuracy": dvae_accuracy, "DVAE_PAR": dvae_PAR, "DVAE_time": dvae_time,
                "DINA_error": dina_error, "DINA_accuracy": dina_accuracy, "DINA_PAR": dina_PAR, "DINA_time": dina_time,
                "DINO_error": dino_error, "DINO_accuracy": dino_accuracy, "DINO_PAR": dino_PAR, "DINO_time": dino_time,
                "GDINA_error": gdina_error, "GDINA_accuracy": gdina_accuracy, "GDINA_PAR": gdina_PAR, "GDINA_time": gdina_time,
                "NPC_error": npc_error, "NPC_accuracy": npc_accuracy, "NPC_PAR": npc_PAR, "NPC_time": npc_time,
            }
            for k, v in best_params.items():
                row[f"param_{k}"] = v

            out_csv = os.path.join(
                RESULTS_DIR,
                f"results_{condition['q_matrix']}_{condition['gate_type']}_{condition['sample_size']}_rep{rep+1}.csv"
            )
            pd.DataFrame([row]).to_csv(out_csv, index=False)
            print(f"Saved replication results to: {out_csv}")

        results.append({"condition": condition, "results": cond_results})

if __name__ == "__main__":
    main()







