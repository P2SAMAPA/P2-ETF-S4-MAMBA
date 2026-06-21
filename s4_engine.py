"""
s4_engine.py — Diagonal State Space Model Engine (DSS / Diagonal S4)
=====================================================================

Theory
------
Diagonal S4 / DSS (Gupta et al. 2022, Smith et al. 2023).

Per-channel IIR filter bank:
    x_k[d] = a[d] · x_{k-1}[d]  +  b[d] · u_k[d]
    y_k[d] = c[d] · x_k[d]      +  D_skip[d] · u_k[d]

a[d] = exp(log_a[d]) ∈ (0,1), HiPPO-inspired log-spaced time constants.

Key optimisation: the scan is vectorised across the ENTIRE BATCH simultaneously.
For a batch of B sequences of length L and D channels:
    x[:, t, :] = a * x[:, t-1, :] + b * U[:, t, :]    ← (B,D) numpy op
This gives B samples for the cost of 1 sequential scan → B× speedup.

References
----------
- Gupta, A., Gu, A. & Berant, J. (2022). Diagonal State Spaces are as Effective
  as Structured State Spaces. NeurIPS 2022.
- Smith, J., Warrington, A. & Linderman, S. (2023). Simplified State Space
  Layers for Sequence Modeling. ICLR 2023.
- Gu, A., Goel, K. & Ré, C. (2021). S4. ICLR 2022.
- Gu, A. & Dao, T. (2023). Mamba. arXiv:2312.00752.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import config


# ── Window-adaptive hyperparameters ──────────────────────────────────────────

def _window_params(window: int) -> Tuple[int, int]:
    """Return (train_stride, n_epochs) scaled to window length."""
    if window <= 126:
        return 5, 60
    elif window <= 252:
        return 10, 50
    elif window <= 504:
        return 15, 40
    else:                   # 1008d
        return 21, 30


# ── HiPPO-inspired diagonal initialisation ───────────────────────────────────

def _hippo_diag_init(D: int, L: int) -> np.ndarray:
    """Log-spaced time constants from 1/L to 1, mapped to log_a (negative)."""
    dt = np.logspace(np.log10(1.0/L), 0.0, D)
    return -dt


# ── DSS Block with BATCHED forward/backward ───────────────────────────────────

class DSSBlock:
    """
    Single Diagonal State Space block — fully batched.

    Parameters (shape D):
        log_a : log decay rates (negative → a ∈ (0,1))
        b     : input gains
        c     : output gains
        D_skip: skip connection
        gamma : LayerNorm scale
        beta  : LayerNorm bias
    """
    def __init__(self, D: int, L: int, rng: np.random.Generator):
        self.D     = D
        self.log_a  = _hippo_diag_init(D, L)
        self.b      = rng.normal(0, 0.1, D)
        self.c      = rng.normal(0, np.sqrt(1.0/D), D)
        self.D_skip = np.ones(D) * 0.1
        self.gamma  = np.ones(D)
        self.beta   = np.zeros(D)

    def forward_batch(self, U: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        U: (B, L, D)
        Returns Y: (B, L, D), cache
        """
        B, L, D = U.shape
        a = np.exp(self.log_a)              # (D,)

        # Batched scan: x[:,t,:] = a*x[:,t-1,:] + b*u[:,t,:]
        X = np.zeros((B, L, D))
        x = np.zeros((B, D))
        for t in range(L):                  # L iters, each O(B*D) numpy op
            x         = a * x + self.b * U[:, t, :]
            X[:, t, :] = x

        Y_ssm = self.c * X + self.D_skip * U    # (B, L, D)

        # LayerNorm over D (last axis)
        mu_ln  = Y_ssm.mean(axis=-1, keepdims=True)    # (B,L,1)
        std_ln = Y_ssm.std(axis=-1,  keepdims=True) + 1e-5
        Y_norm = (Y_ssm - mu_ln) / std_ln
        Y_ln   = self.gamma * Y_norm + self.beta

        Y_out  = Y_ln + U   # residual

        cache = {"U": U, "a": a, "X": X, "Y_ssm": Y_ssm,
                 "mu_ln": mu_ln, "std_ln": std_ln, "Y_norm": Y_norm}
        return Y_out, cache

    def backward_batch(self, dY_out: np.ndarray,
                       cache: dict) -> Tuple[dict, np.ndarray]:
        """
        dY_out: (B, L, D)
        Returns grads, dU: (B, L, D)
        """
        U      = cache["U"]
        a      = cache["a"]
        X      = cache["X"]
        Y_ssm  = cache["Y_ssm"]
        mu_ln  = cache["mu_ln"]
        std_ln = cache["std_ln"]
        Y_norm = cache["Y_norm"]
        B, L, D = U.shape

        dU_res = dY_out.copy()

        # LayerNorm backward
        dY_norm = dY_out * self.gamma
        d_gamma = (dY_out * Y_norm).sum(axis=(0,1))   # (D,)
        d_beta  = dY_out.sum(axis=(0,1))               # (D,)
        dY_ssm  = (dY_norm
                   - dY_norm.mean(axis=-1, keepdims=True)
                   - Y_norm * (dY_norm * Y_norm).mean(axis=-1, keepdims=True)
                  ) / std_ln                           # (B,L,D)

        # SSM output: Y_ssm = c*X + D_skip*U
        dX      = dY_ssm * self.c                      # (B,L,D)
        d_c     = (dY_ssm * X).sum(axis=(0,1))         # (D,)
        d_Dskip = (dY_ssm * U).sum(axis=(0,1))         # (D,)
        dU_skip = dY_ssm * self.D_skip                  # (B,L,D)

        # Batched scan backward
        d_log_a = np.zeros(D)
        d_b     = np.zeros(D)
        dU_ssm  = np.zeros((B, L, D))
        dx_next = np.zeros((B, D))

        X_prev      = np.zeros((B, L, D))
        X_prev[:, 1:, :] = X[:, :-1, :]

        for t in reversed(range(L)):
            dx_t     = dX[:, t, :] + dx_next * a        # (B,D)
            d_log_a += (dx_t * a * X_prev[:, t, :]).sum(axis=0)
            d_b     += (dx_t * U[:, t, :]).sum(axis=0)
            dU_ssm[:, t, :] = dx_t * self.b
            dx_next  = dx_t

        dU = dU_res + dU_skip + dU_ssm

        grads = {"log_a": d_log_a, "b": d_b, "c": d_c,
                 "D_skip": d_Dskip, "gamma": d_gamma, "beta": d_beta}
        return grads, dU

    def params(self) -> dict:
        return {"log_a": self.log_a, "b": self.b, "c": self.c,
                "D_skip": self.D_skip, "gamma": self.gamma, "beta": self.beta}


# ── Full model ────────────────────────────────────────────────────────────────

class DSSModel:
    def __init__(self, in_dim: int, L: int, rng: np.random.Generator):
        D = config.S4_D_MODEL
        self.W_emb  = rng.normal(0, np.sqrt(2.0/in_dim), (D, in_dim))
        self.b_emb  = np.zeros(D)
        self.blocks  = [DSSBlock(D, L, rng) for _ in range(config.S4_N_LAYERS)]
        self.W_head  = rng.normal(0, np.sqrt(1.0/D), (1, D))
        self.b_head  = np.zeros(1)

    def forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        X: (B, L, in_dim) → predictions (B,), caches
        """
        B, L, _ = X.shape
        H = X @ self.W_emb.T + self.b_emb     # (B, L, D)
        caches = [{"X_in": X, "H": H}]
        for block in self.blocks:
            H, cache = block.forward_batch(H)
            caches.append(cache)
        h_last = H[:, -1, :]                   # (B, D)
        preds  = (h_last @ self.W_head.T + self.b_head).ravel()  # (B,)
        caches.append({"h_last": h_last})
        return preds, caches

    def backward_batch(self, d_loss: np.ndarray,
                       caches: list) -> list:
        """
        d_loss: (B,) — per-sample loss gradient
        """
        B  = len(d_loss)
        D  = config.S4_D_MODEL
        L  = caches[0]["H"].shape[1]
        all_grads = []

        # Head backward
        h_last  = caches[-1]["h_last"]               # (B, D)
        dW_head = (d_loss[:, None] * h_last).mean(axis=0, keepdims=True) # (1,D) mean over B? No — sum then divide
        dW_head = (d_loss[:, None] * h_last).T.mean(axis=1, keepdims=True).T  # keep (1,D)
        db_head = d_loss.mean(keepdims=True)
        dh_last = d_loss[:, None] * self.W_head       # (B, D)
        all_grads.append(("head", {"W_head": dW_head, "b_head": db_head}))

        # Build (B, L, D) grad tensor — only last timestep has grad from head
        dH = np.zeros((B, L, D))
        dH[:, -1, :] = dh_last

        # Blocks backward
        for i in reversed(range(len(self.blocks))):
            block_grads, dH = self.blocks[i].backward_batch(dH, caches[i+1])
            all_grads.append((f"block_{i}", block_grads))

        # Embedding backward
        X_in   = caches[0]["X_in"]                  # (B, L, in_dim)
        dW_emb = np.einsum("bld,blk->dk", dH, X_in) / B   # (D, in_dim)
        db_emb = dH.mean(axis=(0,1))
        all_grads.append(("emb", {"W_emb": dW_emb, "b_emb": db_emb}))

        return all_grads

    def all_named_params(self) -> List[Tuple[str, np.ndarray]]:
        params = [("emb_W", self.W_emb), ("emb_b", self.b_emb)]
        for i, block in enumerate(self.blocks):
            for k, v in block.params().items():
                params.append((f"b{i}_{k}", v))
        params += [("head_W", self.W_head), ("head_b", self.b_head)]
        return params

    def apply_adam(self, all_grads, ms, vs, step,
                   lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        gmap = {}
        for name, gd in all_grads:
            if name == "head":
                gmap["head_W"] = gd["W_head"]
                gmap["head_b"] = gd["b_head"]
            elif name == "emb":
                gmap["emb_W"] = gd["W_emb"]
                gmap["emb_b"] = gd["b_emb"]
            else:
                idx = int(name.split("_")[1])
                for k, g in gd.items():
                    gmap[f"b{idx}_{k}"] = g

        for pname, param in self.all_named_params():
            g = gmap.get(pname)
            if g is None or g.shape != param.shape:
                continue
            m = ms[pname]; v = vs[pname]
            m[:] = b1*m + (1-b1)*g
            v[:] = b2*v + (1-b2)*g**2
            m_hat = m / (1-b1**step)
            v_hat = v / (1-b2**step)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        for block in self.blocks:
            np.clip(block.log_a, -10, -1e-3, out=block.log_a)


def _init_adam(model: DSSModel) -> Tuple[dict, dict]:
    ms = {n: np.zeros_like(p) for n, p in model.all_named_params()}
    vs = {n: np.zeros_like(p) for n, p in model.all_named_params()}
    return ms, vs


# ── Data preparation ──────────────────────────────────────────────────────────

def _build_sequences(log_ret: np.ndarray, macro_norm: np.ndarray,
                     window: int, stride: int) -> Tuple[List[np.ndarray], np.ndarray]:
    L = len(log_ret)
    seqs, tgts = [], []
    t = window
    while t + config.PRED_HORIZON <= L:
        seq_ret = log_ret[t-window:t].reshape(-1, 1)
        seq_mac = macro_norm[t-window:t]
        seq     = np.concatenate([seq_ret, seq_mac], axis=1).astype(np.float64)
        fwd     = log_ret[t:t+config.PRED_HORIZON].mean()
        if not np.isnan(fwd) and not np.isnan(seq).any():
            seqs.append(seq)
            tgts.append(fwd)
        t += stride
    return seqs, np.array(tgts)


# ── Training ──────────────────────────────────────────────────────────────────

def _train_dss(seqs: List[np.ndarray], tgts: np.ndarray,
               in_dim: int, window: int,
               rng: np.random.Generator) -> Tuple[DSSModel, float, float]:
    model  = DSSModel(in_dim, window, rng)
    ms, vs = _init_adam(model)
    N      = len(seqs)
    B      = min(config.BATCH_SIZE, N)

    tgt_mu  = tgts.mean()
    tgt_std = tgts.std() + 1e-8
    tgts_n  = (tgts - tgt_mu) / tgt_std

    _, n_epochs = _window_params(window)
    step        = 0
    best_loss   = np.inf
    patience    = 0

    for epoch in range(n_epochs):
        idx        = rng.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, B):
            bi = idx[i:i+B]
            if len(bi) < 2:
                continue

            # Stack batch: (B, L, in_dim)
            X_batch = np.stack([seqs[j] for j in bi], axis=0)
            y_batch = tgts_n[bi]

            # Batched forward
            preds, caches = model.forward_batch(X_batch)

            # Loss + gradient
            resid   = preds - y_batch              # (B,)
            loss    = float(np.mean(resid**2))
            d_loss  = 2.0 * resid / len(bi)        # (B,)

            # Batched backward
            all_grads = model.backward_batch(d_loss, caches)

            step += 1
            model.apply_adam(all_grads, ms, vs, step, lr=config.LR)
            epoch_loss += loss
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}  loss={avg_loss:.6f}")

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience  = 0
        else:
            patience += 1
            if patience >= 8:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return model, tgt_mu, tgt_std


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_ssm_scores(prices: pd.DataFrame, macro_df: pd.DataFrame,
                       tickers: List[str], window: int) -> pd.Series:
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + config.MIN_SEQ_LEN
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu       = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std      = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = macro_vals

    in_dim     = 1 + macro_norm.shape[1]
    stride, _  = _window_params(window)
    rng        = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        mac     = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        seqs, tgts = _build_sequences(log_ret, mac, window, stride)
        if len(seqs) < 4:
            print(f"    {ticker}: only {len(seqs)} samples, skipping")
            continue

        print(f"    Training DSS for {ticker} "
              f"(N={len(seqs)}, L={window}, stride={stride}, in_dim={in_dim})")

        try:
            model, tgt_mu, tgt_std = _train_dss(seqs, tgts, in_dim, window, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            import traceback; traceback.print_exc()
            continue

        today_ret = log_ret[-window:].reshape(-1, 1)
        today_mac = mac[-window:]
        today_seq = np.concatenate([today_ret, today_mac], axis=1).astype(np.float64)

        if np.isnan(today_seq).any():
            continue

        try:
            # Inference: wrap in batch of 1
            preds, _ = model.forward_batch(today_seq[None])
            pred = float(preds[0]) * tgt_std + tgt_mu
            pred = np.clip(pred, -5*tgt_std, 5*tgt_std)
            raw_scores[ticker] = pred
        except Exception as e:
            print(f"    Inference failed {ticker}: {e}")
            continue

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
