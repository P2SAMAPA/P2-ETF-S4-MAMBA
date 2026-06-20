"""
trainer.py — S4/Mamba Engine orchestrator (parallel-window mode)
================================================================

Invocation
----------
  python trainer.py --window 126   # run one window, write shard
  python trainer.py --merge        # merge all shards into final JSON files
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import config
import data_manager
import push_results
from s4_engine import compute_ssm_scores


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


# ── Single-window job ─────────────────────────────────────────────────────────

def run_window(win: int):
    if not config.HF_TOKEN:
        print("HF_TOKEN not set"); return

    df       = data_manager.load_master_data()
    macro_df = data_manager.prepare_macro(df)
    today    = datetime.now().strftime("%Y-%m-%d")
    shard    = {"run_date": today, "window": win, "universes": {}}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name}  window={win}d  "
              f"model={config.MODEL_VARIANT.upper()} ===")

        prices            = data_manager.prepare_prices(df, tickers)
        available_tickers = [t for t in tickers if t in prices.columns]

        if not available_tickers or prices.empty:
            print("  No price data")
            shard["universes"][universe_name] = {}
            continue

        min_bars = win + config.PRED_HORIZON + config.MIN_SEQ_LEN
        if len(prices) < min_bars:
            print(f"  Skipping — need {min_bars} bars, have {len(prices)}")
            shard["universes"][universe_name] = {}
            continue

        try:
            scores = compute_ssm_scores(
                prices   = prices,
                macro_df = macro_df,
                tickers  = available_tickers,
                window   = win,
            )
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback; traceback.print_exc()
            shard["universes"][universe_name] = {}
            continue

        if scores.empty:
            print("  No scores")
            shard["universes"][universe_name] = {}
            continue

        score_dict    = {t: float(s) for t, s in scores.items() if not np.isnan(s)}
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3: {[t for t, _ in sorted_scores[:3]]}")

        shard["universes"][universe_name] = {
            "top_etfs": [
                {"ticker": t, "ssm_score": float(s)}
                for t, s in sorted_scores[:config.TOP_N]
            ],
            "full_ranking": [[t, float(s)] for t, s in sorted_scores],
        }

    Path("shards").mkdir(exist_ok=True)
    shard_path = Path(f"shards/shard_{win}.json")
    with open(shard_path, "w") as f:
        json.dump(convert_to_serializable(shard), f, indent=2)
    print(f"\nWrote {shard_path}")


# ── Merge job ─────────────────────────────────────────────────────────────────

def merge_shards():
    shard_dir   = Path("shards")
    shard_files = sorted(shard_dir.glob("shard_*.json"))
    if not shard_files:
        print("No shards found"); return

    today  = datetime.now().strftime("%Y-%m-%d")
    shards = []
    for sf in shard_files:
        with open(sf) as f:
            shards.append(json.load(f))
        print(f"Loaded {sf.name}  (window={shards[-1]['window']})")

    # Tab 2: per-window
    tab2_universes = {u: {"windows": {}, "run_date": today}
                      for u in config.UNIVERSES}
    for shard in shards:
        win = str(shard["window"])
        for u, u_data in shard["universes"].items():
            if u_data:
                tab2_universes[u]["windows"][win] = u_data

    # Tab 1: best window per ETF
    tab1_universes = {}
    for universe_name in config.UNIVERSES:
        best_per_etf: dict[str, tuple[float, int]] = {}
        for shard in shards:
            win    = int(shard["window"])
            u_data = shard["universes"].get(universe_name, {})
            for ticker, score in u_data.get("full_ranking", []):
                if ticker not in best_per_etf or abs(score) > abs(best_per_etf[ticker][0]):
                    best_per_etf[ticker] = (float(score), win)

        if not best_per_etf:
            tab1_universes[universe_name] = {"top_etfs": [], "full_scores": {}, "run_date": today}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        tab1_universes[universe_name] = {
            "top_etfs": [
                {"ticker": t, "ssm_score": float(s), "best_window": int(w)}
                for t, (s, w) in sorted_etfs[:config.TOP_N]
            ],
            "full_scores": {
                t: {"score": float(s), "best_window": int(w)}
                for t, (s, w) in sorted_etfs
            },
            "run_date": today,
        }
        print(f"  {universe_name} top 3: "
              f"{[e['ticker'] for e in tab1_universes[universe_name]['top_etfs']]}")

    Path("results").mkdir(exist_ok=True)

    tab1_path = Path(f"results/s4_mamba_engine_{today}.json")
    with open(tab1_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": tab1_universes}), f, indent=2)

    tab2_path = Path(f"results/s4_mamba_engine_windows_{today}.json")
    with open(tab2_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": tab2_universes}), f, indent=2)

    print(f"\nWrote {tab1_path.name} and {tab2_path.name}")
    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)
    print("=== Merge complete ===")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--window", type=int, help="Run one window (e.g. --window 252)")
    group.add_argument("--merge",  action="store_true", help="Merge shards and upload")
    args = parser.parse_args()

    if args.merge:
        merge_shards()
    else:
        run_window(args.window)
