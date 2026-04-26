import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from forge.optimizer import OptimizeResult
from forge.profiler import ProfileResult


console = Console()


def render_report(result: OptimizeResult) -> None:
    b = result.baseline
    o = result.optimized

    table = Table(title="FORGE — Benchmark Report", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Optimized", justify="right", style="green")

    def _row(name: str, bval: float, oval: float, fmt: str = ".0f", suffix: str = "") -> None:
        table.add_row(name, f"{bval:{fmt}}{suffix}", f"{oval:{fmt}}{suffix}")

    _row("Step time p50", b.step_time_p50 * 1000, o.step_time_p50 * 1000, ".0f", "ms")
    _row("Step time p90", b.step_time_p90 * 1000, o.step_time_p90 * 1000, ".0f", "ms")
    _row("Tokens/sec", b.tokens_per_sec, o.tokens_per_sec, ".0f")
    _row("MFU", b.mfu * 100, o.mfu * 100, ".1f", "%")
    _row("GPU idle %", b.gpu_idle_pct, o.gpu_idle_pct, ".1f", "%")
    _row("Memory headroom", b.memory_headroom_pct, o.memory_headroom_pct, ".1f", "%")

    console.print(table)
    console.print()
    console.print("[bold]Fix attribution:[/bold]")

    for i, fix in enumerate(result.fixes, 1):
        marker = "  ← largest gain" if fix.fix_name == result.largest_gain_fix else ""
        console.print(
            f"  [{i}] {fix.fix_name:<22} "
            f"[cyan]{fix.tokens_per_sec_before:.0f}[/cyan] → "
            f"[green]{fix.tokens_per_sec_after:.0f}[/green] tok/s  "
            f"[yellow]{fix.delta_pct:+.0f}%[/yellow]{marker}"
        )

    speedup = o.tokens_per_sec / b.tokens_per_sec if b.tokens_per_sec > 0 else 0.0
    console.print()
    console.print(f"[bold green]Total speedup: {speedup:.1f}×[/bold green]")


def save_json(result: OptimizeResult, path: Path) -> Path:
    """Write OptimizeResult to JSON at path. Creates parent dirs. Overwrites if exists."""
    def _prof(p: ProfileResult) -> dict[str, float]:
        return {
            "tokens_per_sec": p.tokens_per_sec,
            "mfu": p.mfu,
            "gpu_idle_pct": p.gpu_idle_pct,
            "step_time_p50_ms": p.step_time_p50 * 1000,
            "step_time_p90_ms": p.step_time_p90 * 1000,
            "memory_headroom_pct": p.memory_headroom_pct,
        }

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": _prof(result.baseline),
        "optimized": _prof(result.optimized),
        "fixes": [
            {
                "name": f.fix_name,
                "tokens_per_sec_before": f.tokens_per_sec_before,
                "tokens_per_sec_after": f.tokens_per_sec_after,
                "delta_pct": f.delta_pct,
            }
            for f in result.fixes
        ],
        "largest_gain_fix": result.largest_gain_fix,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def save_to_db(result: OptimizeResult, db_path: Path = Path("results/forge.db")) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    speedup = (result.optimized.tokens_per_sec / result.baseline.tokens_per_sec
               if result.baseline.tokens_per_sec > 0 else 0.0)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                baseline_tps REAL,
                optimized_tps REAL,
                speedup REAL,
                largest_gain_fix TEXT,
                baseline_mfu REAL,
                optimized_mfu REAL,
                baseline_gpu_idle REAL,
                optimized_gpu_idle REAL
            )
        """)
        conn.execute(
            "INSERT INTO runs (timestamp, baseline_tps, optimized_tps, speedup, largest_gain_fix,"
            " baseline_mfu, optimized_mfu, baseline_gpu_idle, optimized_gpu_idle)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                result.baseline.tokens_per_sec,
                result.optimized.tokens_per_sec,
                speedup,
                result.largest_gain_fix,
                result.baseline.mfu,
                result.optimized.mfu,
                result.baseline.gpu_idle_pct,
                result.optimized.gpu_idle_pct,
            ),
        )
