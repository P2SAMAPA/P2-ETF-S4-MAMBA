"""
s4_engine.py — Diagonal State Space Model Engine (DSS / Diagonal S4)
=====================================================================

Theory
------
**Diagonal S4 / DSS (Gupta et al. 2022, Smith et al. 2023)**

Gupta et al. (2022) proved that a *diagonal* state matrix A is as expressive
as the full structured HiPPO matrix for sequence modelling in practice.
Smith et al. (2023) further simplified this into S5 / simplified SSM.

The diagonal SSM per channel d:

    x_k[d] = a[d] · x_{k-1}[d]  +  b[d] · u_k[d]
    y_k[d] = c[d] · x_k[d]      +  D_skip[d] · u_k[d]

Where:
    a[d] = exp(log_a[d])   ∈ (0,1)   — decay rate, log-parameterised for stability
    b[d]                              — input gain
    c[d]                              — output gain
    D_skip[d]                         — skip connection

This is a **multi-channel causal exponential smoother** — equivalent to a
bank of IIR filters with learnable time constants.

Key properties:
- **No python loops inside the scan** — each timestep is a pure numpy vector op
  O(D) per step, O(LD) total — fast and cache-friendly
- **HiPPO-inspired initialisation** — log_a initialised to give time constants
  spanning 1 to L days (log-spaced), mimicking HiPPO's polynomial basis
- **Analytically differentiable** — clean backprop through the recurrence
- **Provably as expressive as S4** (Gupta et al. 2022)

Architecture
------------

    Input (L, in_dim)
        ↓  Linear embedding  [W_emb, b_emb]
    (L, D)  ← D = S4_D_MODEL
        ↓  DSS Block 1:  diagonal SSM + skip + LayerNorm + residual
    (L, D)
        ↓  DSS Block 2
    (L, D)
        ↓  Last timestep → (D,)
        ↓  Linear head  [W_head, b_head]
    Scalar prediction

Each DSS Block contains:
    - Diagonal SSM (per-channel IIR filter bank): parameters (log_a, b, c, D_skip)
    - LayerNorm: (gamma, beta)
    - Residual connection

Training: Adam with analytical backprop (no finite differences).

References
----------
- Gupta, A., Gu, A. & Berant, J. (2022). Diagonal State Spaces are as Effective
  as Structured State Spaces. NeurIPS 2022.
- Smith, J., Warrington, A. & Linderman, S. (2023). Simplified State Space
  Layers for Sequence Modeling. ICLR 2023.
- Gu, A., Goel, K. & Ré, C. (2021). Efficiently Modeling Long Sequences with
  Structured State Spaces. ICLR 2022.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces. arXiv:2312.00752.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import config


# ── HiPPO-inspired initialisation for diagonal decay ─────────────────────────

def _hippo_diag_init(D: int, L: int) -> np.ndarray:
    """
    Initialise diagonal log_a so that time constants are log-spaced from
    1 to L days. This mimics the multi-scale memory of HiPPO without the
    full lower-triangular matrix.

    a[d] = exp(-dt[d])  where dt[d] is log-spaced in [1/L, 1]
    → log_a[d] = -dt[d]
    """
    dt = np.logspace(np.log10(1.0/L), 0.0, D)  # log-spaced in [1/L, 1]
    return -dt   # log_a < 0 ensures a = exp(log_a) ∈ (0,1)


# ── DSS Block ─────────────────────────────────────────────────────────────────

class DSSBlock:
    """
    Single Diagonal State Space block.

    Parameters (all shape (D,)):
        log_a : log of decay rates  (kept negative for stability)
        b     : input gains
        c     : output gains
        D_skip: skip connection weights
        gamma : LayerNorm scale
        beta  : LayerNorm bias
    """
    def __init__(self, D: int, L: int, rng: np.random.Generator):
        self.D = D
        # Diagonal SSM params
        self.log_a  = _hippo_diag_init(D, L)           # (D,) negative
        self.b      = rng.normal(0, 0.1, D)             # (D,)
        self.c      = rng.normal(0, np.sqrt(1.0/D), D)  # (D,)
        self.D_skip = np.ones(D) * 0.1                  # (D,)
        # LayerNorm
        self.gamma  = np.ones(D)
        self.beta   = np.zeros(D)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, U: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        U: (L, D)
        Returns Y: (L, D), cache
        """
        L, D = U.shape
        a = np.exp(self.log_a)          # (D,) ∈ (0,1)

        # Diagonal SSM scan — pure numpy, no inner python loop over D
        X = np.zeros((L, D))
        x = np.zeros(D)
        for t in range(L):              # O(L) iterations, each O(D) numpy op
            x    = a * x + self.b * U[t]
            X[t] = x

        # SSM output + skip
        Y_ssm = self.c * X + self.D_skip * U    # (L, D)

        # LayerNorm
        mu_ln  = Y_ssm.mean(axis=-1, keepdims=True)
        std_ln = Y_ssm.std(axis=-1, keepdims=True) + 1e-5
        Y_norm = (Y_ssm - mu_ln) / std_ln
        Y_ln   = self.gamma * Y_norm + self.beta

        # Residual
        Y_out  = Y_ln + U               # (L, D)

        cache = {
            "U": U, "a": a, "X": X,
            "Y_ssm": Y_ssm, "mu_ln": mu_ln, "std_ln": std_ln,
            "Y_norm": Y_norm, "Y_ln": Y_ln,
        }
        return Y_out, cache

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self, dY_out: np.ndarray,
                 cache: dict) -> Tuple[dict, np.ndarray]:
        """
        dY_out: (L, D)
        Returns grads dict, dU: (L, D)
        """
        U      = cache["U"]
        a      = cache["a"]
        X      = cache["X"]
        Y_ssm  = cache["Y_ssm"]
        mu_ln  = cache["mu_ln"]
        std_ln = cache["std_ln"]
        Y_norm = cache["Y_norm"]
        L, D   = U.shape

        # Residual: dU gets dY_out directly
        dU_res = dY_out.copy()

        # LayerNorm backward
        dY_ln   = dY_out                                     # (L, D) after residual split
        d_gamma = (dY_ln * Y_norm).sum(axis=0)              # (D,)
        d_beta  = dY_ln.sum(axis=0)                         # (D,)
        dY_norm = dY_ln * self.gamma                         # (L, D)

        # LayerNorm input backward
        dY_ssm  = (dY_norm
                   - dY_norm.mean(axis=-1, keepdims=True)
                   - Y_norm * (dY_norm * Y_norm).mean(axis=-1, keepdims=True)
                  ) / std_ln                                 # (L, D)

        # SSM output: Y_ssm = c*X + D_skip*U
        dX      = dY_ssm * self.c                           # (L, D)
        d_c     = (dY_ssm * X).sum(axis=0)                 # (D,)
        d_Dskip = (dY_ssm * U).sum(axis=0)                 # (D,)
        dU_skip = dY_ssm * self.D_skip                      # (L, D)

        # SSM scan backward: x_k = a*x_{k-1} + b*u_k
        # dL/dx_k = dX[k] + a * dL/dx_{k+1}
        d_log_a = np.zeros(D)
        d_b     = np.zeros(D)
        dU_ssm  = np.zeros((L, D))
        dx_next = np.zeros(D)

        # Reconstruct x_{k-1} from scan (shift X by 1)
        X_prev = np.zeros((L, D))
        X_prev[1:] = X[:-1]

        for t in reversed(range(L)):
            dx_t     = dX[t] + dx_next * a     # chain through recurrence
            d_log_a += dx_t * a * X_prev[t]    # da/d(log_a) = a
            d_b     += dx_t * U[t]
            dU_ssm[t] = dx_t * self.b
            dx_next   = dx_t

        dU = dU_res + dU_skip + dU_ssm

        grads = {
            "log_a": d_log_a, "b": d_b, "c": d_c,
            "D_skip": d_Dskip, "gamma": d_gamma, "beta": d_beta,
        }
        return grads, dU

    def params(self) -> dict:
        return {
            "log_a": self.log_a, "b": self.b, "c": self.c,
            "D_skip": self.D_skip, "gamma": self.gamma, "beta": self.beta,
        }


# ── Full model ────────────────────────────────────────────────────────────────

class DSSModel:
    """
    Stacked DSS blocks with linear input embedding and output head.
    """
    def __init__(self, in_dim: int, L: int, rng: np.random.Generator):
        D = config.S4_D_MODEL
        sc = np.sqrt(2.0 / in_dim)
        self.W_emb  = rng.normal(0, sc, (D, in_dim))
        self.b_emb  = np.zeros(D)
        self.blocks  = [DSSBlock(D, L, rng) for _ in range(config.S4_N_LAYERS)]
        self.W_head  = rng.normal(0, np.sqrt(1.0/D), (1, D))
        self.b_head  = np.zeros(1)

    def forward(self, X: np.ndarray) -> Tuple[float, list]:
        """X: (L, in_dim) → scalar, caches"""
        H = X @ self.W_emb.T + self.b_emb      # (L, D)
        caches = [{"X_in": X, "H": H}]
        for block in self.blocks:
            H, cache = block.forward(H)
            caches.append(cache)
        # Last timestep → head
        h_last = H[-1]                           # (D,)
        out    = float((h_last @ self.W_head.T + self.b_head)[0])
        caches.append({"h_last": h_last})
        return out, caches

    def backward(self, d_loss: float, caches: list) -> list:
        """Returns list of (name, grads_dict)."""
        all_grads = []
        D = config.S4_D_MODEL
        L = caches[0]["H"].shape[0]

        # Head backward
        h_last    = caches[-1]["h_last"]
        dW_head   = np.array([[d_loss]]) * h_last[None, :]   # (1, D)
        db_head   = np.array([d_loss])
        dh_last   = d_loss * self.W_head.ravel()              # (D,)
        all_grads.append(("head", {"W_head": dW_head, "b_head": db_head}))

        # Backprop: dh_last only affects the last timestep
        dH = np.zeros((L, D))
        dH[-1] = dh_last

        # DSS blocks backward (reverse)
        for i in reversed(range(len(self.blocks))):
            block_grads, dH = self.blocks[i].backward(dH, caches[i+1])
            all_grads.append((f"block_{i}", block_grads))

        # Embedding backward
        dW_emb = dH.T @ caches[0]["X_in"]    # (D, in_dim)
        db_emb = dH.sum(axis=0)
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
        # Build grad map
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
            m_hat = m / (1 - b1**step)
            v_hat = v / (1 - b2**step)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Clip log_a to stay negative (stability)
        for i, block in enumerate(self.blocks):
            np.clip(block.log_a, -10, -1e-3, out=block.log_a)


def _init_adam(model: DSSModel) -> Tuple[dict, dict]:
    ms = {n: np.zeros_like(p) for n, p in model.all_named_params()}
    vs = {n: np.zeros_like(p) for n, p in model.all_named_params()}
    return ms, vs


# ── Data preparation ──────────────────────────────────────────────────────────

def _build_sequences(log_ret: np.ndarray, macro_norm: np.ndarray,
                     window: int) -> Tuple[List[np.ndarray], np.ndarray]:
    L      = len(log_ret)
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
        t += config.TRAIN_STRIDE
    return seqs, np.array(tgts)


# ── Training ──────────────────────────────────────────────────────────────────

def _train_dss(seqs: List[np.ndarray], tgts: np.ndarray,
               in_dim: int, window: int,
               rng: np.random.Generator) -> Tuple["DSSModel", float, float]:
    model  = DSSModel(in_dim, window, rng)
    ms, vs = _init_adam(model)
    N      = len(seqs)
    B      = min(config.BATCH_SIZE, N)
    step   = 0

    tgt_mu  = tgts.mean()
    tgt_std = tgts.std() + 1e-8
    tgts_n  = (tgts - tgt_mu) / tgt_std

    best_loss = np.inf
    patience  = 0

    for epoch in range(config.N_EPOCHS):
        idx        = rng.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, B):
            bi = idx[i:i+B]
            if len(bi) < 2:
                continue

            batch_grads = None
            batch_loss  = 0.0

            for j in bi:
                pred, caches = model.forward(seqs[j])
                resid = pred - float(tgts_n[j])
                batch_loss += resid**2

                all_grads = model.backward(2.0 * resid / len(bi), caches)

                if batch_grads is None:
                    batch_grads = all_grads
                else:
                    for ki in range(len(all_grads)):
                        _, gacc = batch_grads[ki]
                        _, gnew = all_grads[ki]
                        for k in gnew:
                            if k in gacc:
                                gacc[k] = gacc[k] + gnew[k]

            step += 1
            model.apply_adam(batch_grads, ms, vs, step, lr=config.LR)
            epoch_loss += batch_loss / len(bi)
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1}/{config.N_EPOCHS}  loss={avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience  = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return model, tgt_mu, tgt_std


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_ssm_scores(prices: pd.DataFrame, macro_df: pd.DataFrame,
                       tickers: List[str], window: int) -> pd.Series:
    """
    Train a DSS model per ETF and return cross-sectional z-scores.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days (sequence length L)

    Returns
    -------
    pd.Series indexed by ticker, values = cross-sectional z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + config.MIN_SEQ_LEN
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro
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

        seqs, tgts = _build_sequences(log_ret, mac, window)
        if len(seqs) < config.BATCH_SIZE:
            print(f"    {ticker}: only {len(seqs)} samples, skipping")
            continue

        print(f"    Training DSS for {ticker} "
              f"(N={len(seqs)}, L={window}, in_dim={in_dim})")

        try:
            model, tgt_mu, tgt_std = _train_dss(seqs, tgts, in_dim, window, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Inference: today's window
        today_ret = log_ret[-window:].reshape(-1, 1)
        today_mac = mac[-window:]
        today_seq = np.concatenate([today_ret, today_mac], axis=1).astype(np.float64)

        if np.isnan(today_seq).any():
            continue

        try:
            pred_norm, _ = model.forward(today_seq)
            pred = float(pred_norm) * tgt_std + tgt_mu
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
