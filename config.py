import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-s4-mamba-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
# S4 shines at long windows — its O(L log L) complexity means 504d/1008d are
# tractable where attention would be O(L²)
WINDOWS = [126, 252, 504, 1008]

# ── S4 / Mamba hyperparameters ────────────────────────────────────────────────

# State space dimension N (hidden state size per channel)
# Higher = more expressive but slower. 16-64 is practical on CPU.
S4_STATE_DIM = 32

# Input channels: log_return + macro signals
# Will be set dynamically from available macro cols
S4_INPUT_DIM = 1      # log_return (macro added as extra channels in engine)

# Number of stacked S4 layers
S4_N_LAYERS = 2

# Model dimension (channel expansion inside each S4 layer)
S4_D_MODEL = 32

# Mamba selective scan parameters
MAMBA_D_STATE  = 16   # SSM state dimension (Mamba uses smaller than S4)
MAMBA_D_CONV   = 4    # local conv width before selective scan
MAMBA_EXPAND   = 2    # channel expansion factor

# Which model variant to use: "s4" or "mamba"
# S4: fixed diagonal HiPPO state matrix — stable long-range
# Mamba: input-dependent (selective) state — better for regime-switching
MODEL_VARIANT = "mamba"

# Training
N_EPOCHS    = 60
LR          = 1e-3
BATCH_SIZE  = 32

# Prediction horizon (forward log-return target)
PRED_HORIZON = 21    # ~1 month

# Minimum sequence length to form one training sample
MIN_SEQ_LEN = 63

# Stride between training samples (step size along time axis)
TRAIN_STRIDE = 5

TOP_N = 3
