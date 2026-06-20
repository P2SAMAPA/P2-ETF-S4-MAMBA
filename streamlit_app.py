import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="S4 / Mamba SSM Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#0d2137; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #1565c0; }
.etf-card    { background:linear-gradient(135deg,#0d2137 0%,#1565c0 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#0a1929 0%,#0d47a1 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

variant = config.MODEL_VARIANT.upper()
st.markdown(f'<div class="main-header">🌀 {variant} Structured State Space Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Gu et al. (2021) S4 · Gu & Dao (2023) Mamba · '
    f'HiPPO state initialisation · O(L log L) long-range sequence modelling · '
    f'Selective scan · Multi-window cross-sectional z-score · '
    f'Active variant: <b>{variant}</b></div>',
    unsafe_allow_html=True)

st.sidebar.markdown(f"## 🌀 {variant} Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Model:** {variant} | **Layers:** {config.S4_N_LAYERS} | "
                    f"**d_model:** {config.S4_D_MODEL}")
st.sidebar.markdown(f"**Epochs:** {config.N_EPOCHS} | **LR:** {config.LR} | "
                    f"**Horizon:** {config.PRED_HORIZON}d")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "s4_mamba_engine_2")
tab2_path = find_latest(files, "s4_mamba_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"🏆 Top ETFs — {variant} State Space Signal")

    with st.expander(f"📖 {variant} Methodology", expanded=True):
        st.markdown(f"""
**{variant}** is a structured state space model that processes sequences via a
linear dynamical system — giving it **O(L log L)** complexity vs O(L²) for attention.
This makes it uniquely suited to long ETF return windows (504d, 1008d).

**Continuous-time state space system:**
```
x'(t) = A x(t) + B u(t)      (state evolution)
y(t)  = C x(t) + D u(t)      (output)
```

**HiPPO initialisation of A:**
```
A_nk = −√((2n+1)(2k+1))  if n>k    (lower triangular)
A_nn = −(n+1)                        (diagonal)
```
Designed to optimally memorise input history via Legendre polynomial projection.

{"**Mamba selective scan:** Δ, B, C are input-dependent — the model selectively focuses or forgets based on each token, making it better for regime-switching markets." if variant == "MAMBA" else "**S4 fixed scan:** A, B, C are fixed parameters — optimal for stationary long-range dependencies."}

| Feature | This engine | LSTM/RNN | Transformer |
|---------|------------|----------|-------------|
| Complexity | O(L log L) | O(L) | O(L²) |
| Long windows | ✅ 1008d+ | ❌ vanishing grad | ❌ too slow |
| Regime switching | {"✅ selective scan" if variant=="MAMBA" else "➖ fixed"} | ✅ gating | ✅ attention |
| Parallelisable | ✅ FFT conv | ❌ sequential | ✅ |
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">SSM score = {etf['ssm_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "SSM Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("SSM Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        f"Gu et al. (2021) S4 / Gu & Dao (2023) Mamba · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header(f"🔍 Explore {variant} Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance — why S4/Mamba excels at long windows", expanded=False):
        st.markdown("""
- **126d** — 6-month sequence; S4 state already compresses more history than LSTM
- **252d** — 1-year sequence; HiPPO projection captures full annual cycle
- **504d** — 2-year sequence; full rate cycle; where S4 > attention computationally
- **1008d** — 4-year sequence; structural macro cycle; O(L²) attention infeasible here
        """)

    st.markdown(f"### {variant} Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">SSM score = {etf['ssm_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "SSM Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
