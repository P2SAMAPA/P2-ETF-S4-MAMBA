# 🌀 P2-ETF-S4-MAMBA

**Structured State Space Sequence Engine — S4 / Mamba**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine applies **Structured State Space Models (S4 / Mamba)** to ETF return
sequences, providing the most efficient long-range dependency modelling in the
suite. Where LSTM vanishes at L=500+ and Transformers scale as O(L²), S4/Mamba
scales as O(L log L) — making 1008-day (4-year) windows fully tractable.

---

## Theory

### S4 — Structured State Space (Gu et al. 2021)

S4 is derived from a continuous-time linear dynamical system:

```
x'(t) = A x(t)  +  B u(t)       (state evolution)
y(t)  = C x(t)  +  D u(t)       (output)
```

Discretised via Zero-Order Hold at step size Δ:

```
Ā = exp(ΔA)
B̄ = A⁻¹(Ā − I)ΔB
x_k = Ā x_{k-1}  +  B̄ u_k
y_k = C x_k  +  D u_k
```

The recurrence can equivalently be computed as a convolution `y = K * u`
where `K_k = C Āᵏ B̄` — enabling **O(L log L) FFT computation**.

### HiPPO Initialisation (Gu et al. 2020)

The state matrix A is initialised with **HiPPO-LegS**:

```
A_{nk} = −√((2n+1)(2k+1))    n > k
A_{nn} = −(n+1)
A_{nk} = 0                    n < k
```

This projects the input history onto Legendre polynomials — designed to
*optimally memorise* the past signal without vanishing or exploding gradients.

### Mamba — Selective SSM (Gu & Dao 2023)

Mamba extends S4 by making Δ, B, C **input-dependent**:

```
Δ(u_t) = softplus(W_Δ · u_t + b_Δ)   ← selective step size
B(u_t) = W_B · u_t                    ← selective input gate
C(u_t) = W_C · u_t                    ← selective output gate
```

This **selective scan** lets the model dynamically focus on or discard
each input token — crucial for financial regime-switching data where some
time periods carry far more information than others.

### Why S4/Mamba for ETFs

| Model | Complexity | L=1008d feasible | Long-range | Regime-aware |
|-------|-----------|-----------------|-----------|-------------|
| LSTM | O(L) | ✅ fast but vanishing grad | ❌ | ✅ gating |
| Transformer | O(L²) | ❌ too slow | ✅ | ✅ attention |
| S4 | O(L log L) | ✅ | ✅ HiPPO | ➖ fixed |
| **Mamba** | **O(L)** | ✅ | ✅ | **✅ selective** |

---

## Input Sequence

Each ETF's input at time t:

```
u_t = [log_return_t,  VIX_t,  DXY_t,  T10Y2Y_t,  IG_SPREAD_t,  HY_SPREAD_t]
```

Standardised per-channel across the training window.

---

## Architecture

```
Input (L, in_dim)
    ↓  Linear embedding
(L, d_model=32)
    ↓  Mamba Layer 1  (selective scan + gating)
(L, d_model)
    ↓  Mamba Layer 2
(L, d_model)
    ↓  Take last timestep  →  (d_model,)
    ↓  Linear head
Scalar prediction (forward log return)
```

**Training:** Adam, 60 epochs, lr=1e-3, analytical backprop (no finite differences)

**Prediction target:** mean log return over next 21 trading days

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MODEL_VARIANT` | `"mamba"` | Switch to `"s4"` for fixed-scan variant |
| `S4_D_MODEL` | 32 | Channel dimension |
| `S4_N_LAYERS` | 2 | Stacked SSM layers |
| `S4_STATE_DIM` | 32 | S4 state size N |
| `MAMBA_D_STATE` | 16 | Mamba SSM state size |
| `MAMBA_EXPAND` | 2 | Inner channel expansion |
| `N_EPOCHS` | 60 | Training epochs |
| `PRED_HORIZON` | 21d | Forward return target |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `126d · 252d · 504d · 1008d`
(deliberately long — this is where S4/Mamba has its largest advantage)

---

## Repository Structure

```
P2-ETF-S4-MAMBA/
├── config.py          # Universes, model variant, hyperparameters
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── s4_engine.py       # Core: HiPPO init, S4 layer, Mamba layer, training
├── trainer.py         # Orchestrator: --window N (shard) or --merge
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Parallel matrix: 4 window jobs + merge
```

---

## Switching Between S4 and Mamba

In `config.py`, change:
```python
MODEL_VARIANT = "mamba"   # or "s4"
```

- **Mamba** — recommended for ETF data: selective scan adapts to macro regime changes
- **S4** — recommended for very long windows (1008d+): fixed HiPPO state is more stable

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-S4-MAMBA
cd P2-ETF-S4-MAMBA
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py --window 252
python trainer.py --merge
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-s4-mamba-results`

---

## References

- Gu, A., Goel, K. & Ré, C. (2021). Efficiently Modeling Long Sequences with
  Structured State Spaces. *ICLR 2022*.
- Gu, A., Johnson, I., Goel, K. et al. (2020). HiPPO: Recurrent Memory with
  Optimal Polynomial Projections. *NeurIPS 2020*.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces. *arXiv:2312.00752*.
- Gupta, A., Gu, A. & Berant, J. (2022). Diagonal State Spaces are as Effective
  as Structured State Spaces. *NeurIPS 2022*.
- Smith, J., Warrington, A. & Linderman, S. (2023). Simplified State Space
  Layers for Sequence Modeling. *ICLR 2023*.
