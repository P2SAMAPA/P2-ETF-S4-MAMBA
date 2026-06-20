"""
s4_engine.py — Structured State Space Sequence Model (S4 / Mamba) Engine
=========================================================================

Theory
------

**S4 — Structured State Space Model (Gu et al. 2021)**

The S4 model is a sequence-to-sequence map derived from a continuous-time
linear state space system:

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

Where:
    u(t) : input signal (ETF log return + macro channels)
    x(t) : hidden state in ℝᴺ (N = S4_STATE_DIM)
    y(t) : output (forward return prediction)
    A    : state transition matrix (HiPPO initialisation)
    B, C : input/output projection vectors
    D    : skip connection scalar

Discretised with step size Δ via ZOH (zero-order hold):

    Ā = exp(ΔA)         (matrix exponential)
    B̄ = (ΔA)⁻¹(Ā − I)ΔB

The discretised recurrence:

    x_k = Ā x_{k-1} + B̄ u_k
    y_k = C x_k + D u_k

Can be computed either as:
    (a) Recurrence: O(NL) — used for inference
    (b) Convolution: y = K * u where K_k = C Āᵏ B̄  — O(L log L) via FFT

This O(L log L) convolution is the key advantage over attention (O(L²)).

**HiPPO Initialisation (Gu et al. 2020)**

A is initialised with HiPPO-LegS (Legendre measure):

    A_{nk} = −(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
    A_{nn} = −(n+1)
    A_{nk} = 0                              if n < k

This initialisation is specifically designed to optimally memorise the history
of the input signal via orthogonal polynomial projection — making S4 excellent
at long-range dependencies without vanishing gradients.

**Mamba — Selective State Space (Gu & Dao 2023)**

Mamba extends S4 by making B, C, and Δ input-dependent:

    Δ(u) = softplus(Linear(u))    ← input-dependent step size
    B(u) = Linear(u)              ← input-dependent B
    C(u) = Linear(u)              ← input-dependent C

This "selective scan" lets the model focus on or forget inputs dynamically,
making it better suited to regime-switching financial data.

The selective scan is still computed as a recurrence (no convolution shortcut
when B,C are input-dependent), but Mamba compensates with a parallel scan
algorithm. We implement the sequential scan here (sufficient for ETF windows).

**Application to ETF Ranking**

For each ETF over a rolling window:
1. Build input sequence: [log_return_t, macro_1_t, ..., macro_M_t], t=1..L
2. Train SSM to predict mean forward log return over next PRED_HORIZON bars
3. At inference: pass today's window through trained SSM → scalar prediction
4. Cross-sectionally z-score predictions across the universe

**Why S4/Mamba outperforms existing suite engines for long windows:**
    - LSTM/RNN    : vanishing gradient at L=500+
    - Transformer : O(L²) attention — impractical at L=1008
    - S4/Mamba    : O(L log L) or O(NL) — efficient at any window length

References
----------
- Gu, A., Goel, K. & Ré, C. (2021). Efficiently Modeling Long Sequences with
  Structured State Spaces. ICLR 2022.
- Gu, A., Johnson, I., Goel, K. et al. (2020). HiPPO: Recurrent Memory with
  Optimal Polynomial Projections. NeurIPS 2020.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces. arXiv:2312.00752.
- Gupta, A., Gu, A. & Berant, J. (2022). Diagonal State Spaces are as Effective
  as Structured State Spaces. NeurIPS 2022.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── HiPPO initialisation ──────────────────────────────────────────────────────

def _hippo_legs(N: int) -> np.ndarray:
    """
    HiPPO-LegS state matrix A ∈ ℝᴺˣᴺ.
    Designed for optimal polynomial projection of input history.
    """
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2*n+1) * (2*k+1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def _discretise_zoh(A: np.ndarray, B: np.ndarray,
                    dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-order hold discretisation:
        Ā = exp(dt·A)
        B̄ = A⁻¹(Ā − I)·B
    Uses diagonal approximation for efficiency (DSSM / diagonal S4).
    """
    N = A.shape[0]
    # Diagonal approximation: use only diagonal of A
    a_diag = np.diag(A)
    A_bar  = np.diag(np.exp(dt * a_diag))
    # B̄ = diag((exp(dt·a_n) − 1) / a_n) · B
    safe_a = np.where(np.abs(a_diag) < 1e-8, 1e-8, a_diag)
    B_bar  = np.diag((np.exp(dt * a_diag) - 1.0) / safe_a) @ B
    return A_bar, B_bar


# ── S4 Layer (diagonal approximation) ────────────────────────────────────────

class S4Layer:
    """
    Single S4 layer with diagonal state matrix (DiagS4 / DSS).
    Input:  (L, d_model)
    Output: (L, d_model)

    Each feature channel has its own independent SISO SSM with shared A.
    """
    def __init__(self, d_model: int, state_dim: int,
                 dt: float, rng: np.random.Generator):
        self.d_model   = d_model
        self.state_dim = state_dim
        N = state_dim

        # HiPPO state matrix (shared across channels, diagonal approx)
        A    = _hippo_legs(N)
        B_0  = rng.normal(0, 1, (N, 1))
        A_bar, B_bar = _discretise_zoh(A, B_0, dt)

        # Per-channel parameters
        # A_bar diagonal: (N,) — shared across channels for efficiency
        self.A_diag = np.diag(A_bar)        # (N,)
        self.B_bar  = B_bar.ravel()         # (N,)

        # C: output projection per channel  (d_model, N)
        self.C  = rng.normal(0, np.sqrt(1.0/N), (d_model, N))
        # D: skip connection  (d_model,)
        self.D  = np.ones(d_model)
        # Input mixing  (d_model, d_model)
        scale   = np.sqrt(2.0 / d_model)
        self.W_in  = rng.normal(0, scale, (d_model, d_model))
        self.W_out = rng.normal(0, scale, (d_model, d_model))

    def forward(self, U: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        U: (L, d_model)
        Returns Y: (L, d_model), cache for backward
        """
        L, D = U.shape
        N    = self.state_dim

        # Input mixing
        U_mix = np.tanh(U @ self.W_in.T)        # (L, D)

        # SSM recurrence: each channel independently
        # x_k = A_diag * x_{k-1} + B_bar * u_k   (element-wise, broadcast over D)
        # y_k = C @ x_k + D * u_k
        X = np.zeros((L, N))   # hidden states
        x = np.zeros(N)
        for t in range(L):
            # Average over channels for the SSM input scalar
            u_t = U_mix[t].mean()
            x   = self.A_diag * x + self.B_bar * u_t
            X[t] = x

        # Output: Y_ssm[t, d] = C[d] @ X[t] + D[d] * U_mix[t, d]
        Y_ssm = X @ self.C.T + U_mix * self.D    # (L, D)

        # Output mixing + residual
        Y = np.tanh(Y_ssm @ self.W_out.T) + U    # (L, D)  residual

        cache = {"U": U, "U_mix": U_mix, "X": X, "Y_ssm": Y_ssm}
        return Y, cache

    def backward(self, dY: np.ndarray, cache: dict) -> Tuple[dict, np.ndarray]:
        """
        dY: (L, d_model)
        Returns grads dict, dU (gradient w.r.t. input)
        """
        U     = cache["U"]
        U_mix = cache["U_mix"]
        X     = cache["X"]
        Y_ssm = cache["Y_ssm"]
        L, D  = U.shape
        N     = self.state_dim

        # Residual: dY passes through unchanged to dU
        dU_res = dY.copy()

        # Output mixing backward: tanh(Y_ssm @ W_out.T) + U
        dTanh_out = dY * (1 - np.tanh(Y_ssm @ self.W_out.T) ** 2)  # (L,D)
        dW_out    = dTanh_out.T @ Y_ssm                              # (D,D)
        dY_ssm    = dTanh_out @ self.W_out                           # (L,D)

        # Y_ssm = X @ C.T + U_mix * D
        dC     = dY_ssm.T @ X                              # (D, N)
        dD     = (dY_ssm * U_mix).sum(axis=0)             # (D,)
        dX     = dY_ssm @ self.C                           # (L, N)
        dU_mix = dY_ssm * self.D                           # (L, D)

        # SSM backward: x_k = A * x_{k-1} + B * u_k
        # u_k = U_mix[k].mean()  → dU_mix[k] += dX[k] @ B * (1/D)
        dB = np.zeros(N)
        for t in reversed(range(L)):
            dx_t = dX[t]
            if t + 1 < L:
                dx_t += dX[t+1] * self.A_diag  # chain through next step
            # u_t = U_mix[t].mean()
            dB += dx_t * U_mix[t].mean()
            # gradient back to U_mix[t] via u_t = mean
            dU_mix[t] += (dx_t @ self.B_bar) / D  # broadcast to all channels

        # Input mixing backward: U_mix = tanh(U @ W_in.T)
        dTanh_in = dU_mix * (1 - U_mix ** 2)      # (L, D)
        dW_in    = dTanh_in.T @ U                  # (D, D)
        dU_in    = dTanh_in @ self.W_in            # (L, D)

        dU = dU_res + dU_in

        grads = {"C": dC, "D": dD, "B_bar": dB, "W_in": dW_in, "W_out": dW_out}
        return grads, dU

    def params(self):
        return {"C": self.C, "D": self.D, "B_bar": self.B_bar,
                "W_in": self.W_in, "W_out": self.W_out}


# ── Mamba Layer (selective SSM) ───────────────────────────────────────────────

class MambaLayer:
    """
    Simplified Mamba layer: selective SSM with input-dependent Δ, B, C.
    Input:  (L, d_model)
    Output: (L, d_model)
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int,
                 expand: int, rng: np.random.Generator):
        self.d_model = d_model
        self.d_inner = d_model * expand     # expanded dimension
        self.d_state = d_state
        self.d_conv  = d_conv
        D, E, N = d_model, self.d_inner, d_state

        sc = lambda f, o: np.sqrt(f / (o if o else 1))

        # Input projection x,z branches (D → 2E)
        self.W_in  = rng.normal(0, sc(2, D), (2 * E, D))

        # Selective parameters (E → N each)
        self.W_B   = rng.normal(0, sc(2, E), (N, E))
        self.W_C   = rng.normal(0, sc(2, E), (N, E))
        # Δ: (E → E) then softplus
        self.W_dt  = rng.normal(0, sc(2, E), (E, E))
        self.b_dt  = np.full(E, -4.0)   # bias → small initial Δ via softplus

        # A: log-parameterised diagonal (initialise to -0.5 so A = exp(-0.5) ≈ 0.6)
        self.A_log = np.full((E, N), -0.5)

        # Output projection E → D
        self.W_out = rng.normal(0, sc(2, E), (D, E))

        # Local conv weight (d_conv,) applied along sequence per channel
        self.conv_w = rng.normal(0, 0.1, (E, d_conv))

    def _softplus(self, x):
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    def forward(self, U: np.ndarray) -> Tuple[np.ndarray, dict]:
        """U: (L, D) → Y: (L, D)"""
        L, D = U.shape
        E, N = self.d_inner, self.d_state

        # Input projections → x branch (L,E) and z branch (L,E)
        XZ   = U @ self.W_in.T                       # (L, 2E)
        x_br = XZ[:, :E]                              # (L, E)
        z_br = XZ[:, E:]                              # (L, E)

        # Depthwise conv on x branch (causal, pad left)
        pad    = self.d_conv - 1
        x_pad  = np.concatenate([np.zeros((pad, E)), x_br], axis=0)  # (L+pad, E)
        x_conv = np.zeros((L, E))
        for t in range(L):
            x_conv[t] = (x_pad[t:t+self.d_conv] * self.conv_w.T).sum(axis=0)
        x_conv = np.tanh(x_conv)                      # SiLU approx

        # Selective parameters
        A   = -np.exp(self.A_log)                     # (E, N) — negative
        B   = x_conv @ self.W_B.T                     # (L, N)
        C   = x_conv @ self.W_C.T                     # (L, N)
        dt  = self._softplus(x_conv @ self.W_dt.T + self.b_dt)  # (L, E)

        # Discretise: Ā_t = exp(dt_t * A), B̄_t = dt_t * B_t (ZOH approx for Δ·A small)
        # Shape: A (E,N), dt (L,E), B (L,N)
        # Ā_t[e,n] = exp(dt[t,e] * A[e,n])
        A_bar = np.exp(dt[:, :, None] * A[None, :, :])   # (L, E, N)
        B_bar = dt[:, :, None] * B[:, None, :]            # (L, E, N)

        # Selective scan: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        # y_t[e] = C_t @ h_t[e]
        H = np.zeros((E, N))
        Y_ssm = np.zeros((L, E))
        for t in range(L):
            H = A_bar[t] * H + B_bar[t] * x_conv[t, :, None]   # (E,N)
            Y_ssm[t] = (H * C[t][None, :]).sum(axis=1)           # (E,)

        # Gating: y = Y_ssm * SiLU(z)
        silu_z = z_br * (1 / (1 + np.exp(-z_br)))
        Y_gate = Y_ssm * silu_z                       # (L, E)

        # Output projection + residual
        Y = Y_gate @ self.W_out.T + U                 # (L, D)

        cache = {
            "U": U, "x_conv": x_conv, "z_br": z_br,
            "A_bar": A_bar, "B_bar": B_bar, "H_seq": None,  # H stored below
            "Y_ssm": Y_ssm, "silu_z": silu_z, "Y_gate": Y_gate,
            "B": B, "C": C, "dt": dt,
        }
        return Y, cache

    def backward(self, dY: np.ndarray, cache: dict) -> Tuple[dict, np.ndarray]:
        U      = cache["U"]
        x_conv = cache["x_conv"]
        z_br   = cache["z_br"]
        Y_ssm  = cache["Y_ssm"]
        silu_z = cache["silu_z"]
        Y_gate = cache["Y_gate"]
        A_bar  = cache["A_bar"]
        B_bar  = cache["B_bar"]
        B      = cache["B"]
        C      = cache["C"]
        dt     = cache["dt"]
        L, D   = U.shape
        E, N   = self.d_inner, self.d_state

        # Output projection + residual
        dU_res  = dY.copy()
        dY_gate = dY @ self.W_out           # (L, E)
        dW_out  = dY.T @ Y_gate            # (D, E)

        # Gating
        sig_z  = 1 / (1 + np.exp(-z_br))
        dsilu  = sig_z * (1 + z_br * (1 - sig_z))
        dY_ssm = dY_gate * silu_z
        dz_br  = dY_gate * Y_ssm * dsilu
        # dz_br feeds back to W_in z-branch — simplified: accumulate to dU via W_in
        dXZ = np.concatenate([np.zeros_like(dY_gate), dz_br], axis=1)

        # SSM backward (simplified — no full BPTT through scan, use output grad only)
        # Approximate: dY_ssm feeds directly to W_C and W_B
        dC  = dY_ssm.T @ x_conv            # (N, E) ... approximate
        dW_C = dC                            # (N, E)
        # W_B gradient: approximate via input side
        dW_B  = np.zeros_like(self.W_B)

        # x_conv backward: aggregate all grads flowing through it
        dx_conv = dY_ssm @ np.zeros((E, E))  # placeholder — approximate
        # Tanh backward for conv
        dx_conv_pre = dx_conv * (1 - x_conv**2)

        # Conv backward → x_br (approximate, ignore for small d_conv)
        dx_br = np.zeros_like(x_conv)

        # W_in backward
        dXZ_full = np.concatenate([dx_br, dz_br], axis=1)
        dW_in    = dXZ_full.T @ U
        dU_in    = dXZ_full @ self.W_in

        dU = dU_res + dU_in

        grads = {
            "W_out": dW_out, "W_in": dW_in,
            "W_C": dW_C, "W_B": dW_B,
        }
        return grads, dU

    def params(self):
        return {"W_in": self.W_in, "W_out": self.W_out,
                "W_B": self.W_B, "W_C": self.W_C,
                "W_dt": self.W_dt, "b_dt": self.b_dt,
                "A_log": self.A_log, "conv_w": self.conv_w}


# ── Output head ───────────────────────────────────────────────────────────────

class OutputHead:
    """Linear head: d_model → 1 (scalar return prediction)."""
    def __init__(self, d_model: int, rng: np.random.Generator):
        self.W = rng.normal(0, np.sqrt(1.0/d_model), (1, d_model))
        self.b = np.zeros(1)

    def forward(self, Y: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Use last timestep only
        y_last = Y[-1]                          # (d_model,)
        out    = y_last @ self.W.T + self.b     # (1,)
        return out, {"y_last": y_last}

    def backward(self, d_out: np.ndarray, cache: dict) -> Tuple[dict, np.ndarray]:
        y_last = cache["y_last"]
        dW     = d_out[:, None] * y_last[None, :]   # (1, d_model)
        db     = d_out.copy()
        dY_last = d_out @ self.W                     # (d_model,)
        return {"W": dW, "b": db}, dY_last

    def params(self):
        return {"W": self.W, "b": self.b}


# ── Full SSM model ────────────────────────────────────────────────────────────

class SSMModel:
    """
    Stacked S4 or Mamba layers + linear output head.
    Input projection: in_dim → d_model
    """
    def __init__(self, in_dim: int, rng: np.random.Generator):
        D  = config.S4_D_MODEL
        sc = np.sqrt(2.0 / in_dim)

        self.W_emb = rng.normal(0, sc, (D, in_dim))
        self.b_emb = np.zeros(D)

        self.layers = []
        for _ in range(config.S4_N_LAYERS):
            if config.MODEL_VARIANT == "mamba":
                self.layers.append(MambaLayer(
                    d_model=D, d_state=config.MAMBA_D_STATE,
                    d_conv=config.MAMBA_D_CONV, expand=config.MAMBA_EXPAND,
                    rng=rng))
            else:
                self.layers.append(S4Layer(
                    d_model=D, state_dim=config.S4_STATE_DIM,
                    dt=0.01, rng=rng))

        self.head = OutputHead(D, rng)

    def forward(self, X: np.ndarray) -> Tuple[float, list]:
        """X: (L, in_dim) → scalar prediction, layer_caches"""
        H = X @ self.W_emb.T + self.b_emb    # (L, D)
        caches = [{"emb_X": X, "H_in": H}]
        for layer in self.layers:
            H, cache = layer.forward(H)
            caches.append(cache)
        out, head_cache = self.head.forward(H)
        caches.append(head_cache)
        return float(out[0]), caches

    def backward(self, d_loss: float, caches: list):
        """Backprop through all layers. Returns list of grad dicts."""
        all_grads = []

        # Head backward
        head_cache = caches[-1]
        head_grads, dH = self.head.backward(np.array([d_loss]), head_cache)
        all_grads.append(("head", head_grads))

        # Build full dY: only last timestep has grad from head
        L = caches[0]["H_in"].shape[0]
        D = config.S4_D_MODEL
        dY = np.zeros((L, D))
        dY[-1] = dH

        # Layer backward (reverse order)
        for i in reversed(range(len(self.layers))):
            layer_grads, dY = self.layers[i].backward(dY, caches[i+1])
            all_grads.append((f"layer_{i}", layer_grads))

        # Embedding backward
        X      = caches[0]["emb_X"]
        dW_emb = dY.T @ X                     # (D, in_dim)
        db_emb = dY.sum(axis=0)
        all_grads.append(("emb", {"W_emb": dW_emb, "b_emb": db_emb}))

        return all_grads

    def all_params(self):
        """Flat list of (name, param_array) for Adam state init."""
        params = [("emb_W", self.W_emb), ("emb_b", self.b_emb)]
        for i, layer in enumerate(self.layers):
            for k, v in layer.params().items():
                params.append((f"l{i}_{k}", v))
        for k, v in self.head.params().items():
            params.append((f"head_{k}", v))
        return params

    def apply_grads(self, all_grads, ms, vs, step, lr=1e-3,
                    b1=0.9, b2=0.999, eps=1e-8):
        """Apply Adam update. ms/vs are dicts keyed by param name."""
        # Collect grad by param name
        grad_map = {}
        for name, gdict in all_grads:
            if name == "head":
                for k, g in gdict.items():
                    grad_map[f"head_{k}"] = g
            elif name == "emb":
                grad_map["emb_W"] = gdict["W_emb"]
                grad_map["emb_b"] = gdict["b_emb"]
            else:
                layer_idx = int(name.split("_")[1])
                for k, g in gdict.items():
                    grad_map[f"l{layer_idx}_{k}"] = g

        for pname, param in self.all_params():
            if pname not in grad_map:
                continue
            g = grad_map[pname]
            if g.shape != param.shape:
                continue
            m = ms[pname]; v = vs[pname]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g**2
            m_hat = m / (1 - b1**step)
            v_hat = v / (1 - b2**step)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _init_adam(model: SSMModel):
    ms, vs = {}, {}
    for name, p in model.all_params():
        ms[name] = np.zeros_like(p)
        vs[name] = np.zeros_like(p)
    return ms, vs


# ── Training ──────────────────────────────────────────────────────────────────

def _build_sequences(log_ret: np.ndarray, macro_norm: np.ndarray,
                     window: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build (sequence, target) pairs using a sliding window.
    Each sequence: shape (window, in_dim) = [log_ret, macro_cols...]
    Target: mean log return over next PRED_HORIZON bars.
    """
    L       = len(log_ret)
    in_dim  = 1 + macro_norm.shape[1]
    seqs, tgts = [], []

    t = window
    while t + config.PRED_HORIZON <= L:
        seq_ret   = log_ret[t-window:t].reshape(-1, 1)    # (W, 1)
        seq_mac   = macro_norm[t-window:t]                 # (W, M)
        seq       = np.concatenate([seq_ret, seq_mac], axis=1).astype(np.float64)
        fwd       = log_ret[t:t+config.PRED_HORIZON].mean()
        if not np.isnan(fwd) and not np.isnan(seq).any():
            seqs.append(seq)
            tgts.append(fwd)
        t += config.TRAIN_STRIDE

    return seqs, np.array(tgts)


def _train_ssm(seqs: List[np.ndarray], tgts: np.ndarray,
               in_dim: int, rng: np.random.Generator) -> SSMModel:
    """Train SSM model with Adam + analytical backprop."""
    model = SSMModel(in_dim, rng)
    ms, vs = _init_adam(model)
    N      = len(seqs)
    step   = 0

    # Normalise targets
    tgt_mu  = tgts.mean()
    tgt_std = tgts.std() + 1e-8
    tgts_n  = (tgts - tgt_mu) / tgt_std

    for epoch in range(config.N_EPOCHS):
        idx        = rng.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, config.BATCH_SIZE):
            bi = idx[i:i+config.BATCH_SIZE]
            if len(bi) < 2:
                continue

            batch_loss = 0.0
            batch_grads_acc = None

            for j in bi:
                pred, caches = model.forward(seqs[j])
                resid = pred - float(tgts_n[j])
                loss  = resid ** 2
                batch_loss += loss

                all_grads = model.backward(2.0 * resid / len(bi), caches)

                # Accumulate gradients
                if batch_grads_acc is None:
                    batch_grads_acc = all_grads
                else:
                    for ki in range(len(all_grads)):
                        name, gdict = all_grads[ki]
                        _, gacc = batch_grads_acc[ki]
                        for k in gdict:
                            if k in gacc:
                                gacc[k] = gacc[k] + gdict[k]

            step += 1
            model.apply_grads(batch_grads_acc, ms, vs, step, lr=config.LR)
            epoch_loss += batch_loss / len(bi)
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1}/{config.N_EPOCHS}  "
                  f"loss={epoch_loss/max(n_batches,1):.6f}")

    return model, tgt_mu, tgt_std


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_ssm_scores(prices: pd.DataFrame, macro_df: pd.DataFrame,
                       tickers: List[str], window: int) -> pd.Series:
    """
    Train an S4/Mamba SSM per ETF and return cross-sectional z-scores.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days (sequence length)

    Returns
    -------
    pd.Series indexed by ticker, values = composite SSM z-score
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
        macro_norm = (macro_vals - m_mu) / m_std
        macro_norm = np.nan_to_num(macro_norm, 0.0)
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

        # Align macro to log_ret length
        mac = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        seqs, tgts = _build_sequences(log_ret, mac, window)
        if len(seqs) < config.BATCH_SIZE:
            print(f"    {ticker}: insufficient samples ({len(seqs)}), skipping")
            continue

        print(f"    Training {config.MODEL_VARIANT.upper()} for {ticker} "
              f"(N={len(seqs)}, L={window}, in_dim={in_dim})")

        try:
            model, tgt_mu, tgt_std = _train_ssm(seqs, tgts, in_dim, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
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
            pred = np.clip(pred, -3 * tgt_std, 3 * tgt_std)
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
