#!/usr/bin/env python3
"""Forge — training efficiency profiler and optimizer for Apple Silicon."""
import argparse
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent / "results"
_DB_PATH = _RESULTS_DIR / "forge.db"


def cmd_baseline(steps: int) -> None:
    from forge.harness import BASELINE_CONFIG
    from forge.profiler import profile
    cfg = replace(BASELINE_CONFIG, n_steps=steps)
    print(f"\nRunning baseline ({steps} steps)...")
    result = profile(cfg)
    print(f"\nBaseline results:")
    print(f"  Tokens/sec:      {result.tokens_per_sec:.0f}")
    print(f"  MFU:             {result.mfu * 100:.1f}%")
    print(f"  GPU idle:        {result.gpu_idle_pct:.1f}%")
    print(f"  Step time p50:   {result.step_time_p50 * 1000:.0f}ms")
    print(f"  Step time p90:   {result.step_time_p90 * 1000:.0f}ms")
    print(f"  Load frac:       {result.load_frac * 100:.0f}%")
    print(f"  Transfer frac:   {result.transfer_frac * 100:.0f}%")
    print(f"  Forward frac:    {result.forward_frac * 100:.0f}%")
    print(f"  Backward frac:   {result.backward_frac * 100:.0f}%")


def cmd_optimize(steps: int) -> None:
    from forge.harness import BASELINE_CONFIG
    from forge.optimizer import optimize
    from forge.report import render_report, save_json, save_to_db
    cfg = replace(BASELINE_CONFIG, n_steps=steps)
    print(f"\nRunning optimize ({steps} steps per phase)...")
    result = optimize(cfg)
    print("\n")
    render_report(result)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = _RESULTS_DIR / f"{ts}.json"
    save_json(result, json_path)
    save_to_db(result, _DB_PATH)
    print(f"\nResults saved to {json_path}")


def cmd_config_search(steps: int) -> None:
    from forge.config_search import config_search, save_optimal_yaml
    print(f"\nRunning config search ({steps} steps per cell)...")
    optimal, all_results = config_search(n_steps=steps)
    save_optimal_yaml(optimal)
    print(f"\nOptimal config: num_workers={optimal.num_workers}, batch_size={optimal.batch_size}")
    print(f"Saved to configs/optimal.yaml\n")
    print(f"  {'workers':>7} {'batch':>6} {'tok/s':>10} {'mem%':>6}")
    print("  " + "-" * 32)
    for r in sorted(all_results, key=lambda x: -x.tokens_per_sec):
        print(f"  {r.num_workers:>7} {r.batch_size:>6} {r.tokens_per_sec:>10.0f} {r.memory_used_pct:>6.0f}")


def cmd_report() -> None:
    if not _DB_PATH.exists():
        print("No results found. Run --optimize first.")
        sys.exit(1)
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT timestamp, baseline_tps, optimized_tps, speedup, largest_gain_fix "
                "FROM runs ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
    except sqlite3.OperationalError:
        print("No runs recorded yet. Run --optimize first.")
        sys.exit(1)
    if not rows:
        print("No runs recorded yet. Run --optimize first.")
        sys.exit(0)
    print(f"\n{'Timestamp':<22} {'Baseline':>10} {'Optimized':>10} {'Speedup':>9} {'Largest gain':>14}")
    print("-" * 70)
    for r in rows:
        print(f"{r[0]:<22} {r[1]:>10.0f} {r[2]:>10.0f} {r[3]:>8.1f}× {r[4]:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forge — training efficiency profiler for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python forge.py --baseline             # profile the broken training loop
  python forge.py --optimize             # apply all fixes, show before/after
  python forge.py --config-search        # grid search optimal DataLoader config
  python forge.py --report               # show run history
  python forge.py --optimize --steps 20  # quick 20-step run
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true",
                       help="Profile baseline (broken) training loop")
    group.add_argument("--optimize", action="store_true",
                       help="Apply all fixes sequentially, show before/after report")
    group.add_argument("--config-search", action="store_true",
                       help="Grid search optimal num_workers × batch_size")
    group.add_argument("--report", action="store_true",
                       help="Show history of benchmark results from SQLite")
    parser.add_argument("--steps", type=int, default=100,
                        help="Training steps per benchmark phase (default: 100)")

    args = parser.parse_args()

    if args.baseline:
        cmd_baseline(args.steps)
    elif args.optimize:
        cmd_optimize(args.steps)
    elif args.config_search:
        cmd_config_search(args.steps)
    elif args.report:
        cmd_report()
    else:
        parser.error("Unhandled command — this is a bug")


if __name__ == "__main__":
    main()
