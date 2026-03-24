#!/usr/bin/env python3
"""sweep_mem_update.py — benchmark MEM_UPDATE_EVERY=[1, 4, 8] vs baseline.

Runs four configurations sequentially in a single script execution:
  1. baseline (TRAIN_WITH_MEMORY=0)
  2. MEM_UPDATE_EVERY=1
  3. MEM_UPDATE_EVERY=4
  4. MEM_UPDATE_EVERY=8

The torchinductor / Triton kernel cache at /tmp/torchinductor_root/ accumulates
across runs, so later runs pay less compilation overhead.

Usage (from repo root with venv active):
    python sweep_mem_update.py |& tee /tmp/sweep_mem_update.log
"""

import os
import re
import subprocess
import sys

# ─── Sweep configuration ────────────────────────────────────────────────────

# val_bpb from the naive-baseline record (used as fallback delta reference).
RECORD_BASELINE_VAL_BPB = 1.2244

CONFIGS = [
    {
        "tag": "baseline",
        "update_every": None,
        "env": {},
    },
    {
        "tag": "mem_update_every=1",
        "update_every": 1,
        "env": {
            "TRAIN_WITH_MEMORY": "1",
            "MEM_INJECT_SCALE": "0.10",
            "MEM_UPDATE_EVERY": "1",
        },
    },
    {
        "tag": "mem_update_every=4",
        "update_every": 4,
        "env": {
            "TRAIN_WITH_MEMORY": "1",
            "MEM_INJECT_SCALE": "0.10",
            "MEM_UPDATE_EVERY": "4",
        },
    },
    {
        "tag": "mem_update_every=8",
        "update_every": 8,
        "env": {
            "TRAIN_WITH_MEMORY": "1",
            "MEM_INJECT_SCALE": "0.10",
            "MEM_UPDATE_EVERY": "8",
        },
    },
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

SEP = "=" * 72


def run_and_capture(tag: str, extra_env: dict) -> str:
    """Spawn train_gpt.py, stream stdout/stderr live, and return captured output."""
    print(f"\n{SEP}", flush=True)
    print(f"  SWEEP RUN: {tag}", flush=True)
    for k, v in sorted(extra_env.items()):
        print(f"    {k}={v}", flush=True)
    print(SEP, flush=True)

    env = os.environ.copy()
    env.update(extra_env)

    proc = subprocess.Popen(
        [sys.executable, "-u", "train_gpt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)

    proc.wait()
    if proc.returncode != 0:
        print(f"  [WARNING] exit code {proc.returncode} for run: {tag}", flush=True)

    return "".join(lines)


def parse_run(output: str) -> dict:
    """Extract key metrics from a completed run's captured output."""
    result: dict = {
        "steps": None,
        "step_avg_ms": None,
        "val_bpb": None,
        "early_stop": False,
    }

    # Most accurate val_bpb: post-quantisation roundtrip.
    m = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\S+)", output
    )
    if m:
        result["val_bpb"] = float(m.group(1))

    # Steps completed (early-stop line is present when wallclock cap is hit).
    m = re.search(r"stopping_early:.*step:(\d+)/\d+", output)
    if m:
        result["early_stop"] = True
        result["steps"] = int(m.group(1))

    # Last validation line: use its step count (if not set above) and step_avg.
    # step_avg on a val line = total_train_ms / steps — a settled, cumulative avg.
    val_lines = re.findall(
        r"step:(\d+)/\d+ val_loss:\S+ val_bpb:\S+ train_time:\S+ step_avg:(\S+)ms",
        output,
    )
    if val_lines:
        last_step, last_avg = val_lines[-1]
        if result["steps"] is None:
            result["steps"] = int(last_step)
        result["step_avg_ms"] = float(last_avg)

    return result


# ─── Main sweep ──────────────────────────────────────────────────────────────

def main() -> None:
    results: list[dict] = []

    for cfg in CONFIGS:
        output = run_and_capture(cfg["tag"], cfg["env"])
        parsed = parse_run(output)
        parsed["tag"] = cfg["tag"]
        parsed["update_every"] = cfg["update_every"]
        results.append(parsed)

    # Prefer the sweep's own baseline val_bpb for delta; fall back to saved record.
    sweep_baseline_bpb = next(
        (r["val_bpb"] for r in results if r["tag"] == "baseline" and r["val_bpb"] is not None),
        None,
    )
    ref_bpb = sweep_baseline_bpb if sweep_baseline_bpb is not None else RECORD_BASELINE_VAL_BPB

    # ─── Summary table ───────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    source = "this sweep" if sweep_baseline_bpb is not None else f"saved record ({RECORD_BASELINE_VAL_BPB})"
    print(f"  SWEEP RESULTS  (ref baseline val_bpb={ref_bpb:.4f} from {source})")
    print(SEP)

    col_tag = 22
    header = (
        f"{'Config':<{col_tag}} | {'step_avg_ms':>11} | {'steps':>5} | "
        f"{'val_bpb':>7} | {'delta_vs_baseline':>18}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        tag     = r["tag"]
        avg_ms  = f"{r['step_avg_ms']:.1f}" if r["step_avg_ms"] is not None else "N/A"
        steps   = str(r["steps"]) if r["steps"] is not None else "N/A"
        bpb     = f"{r['val_bpb']:.4f}" if r["val_bpb"] is not None else "N/A"
        early   = "*" if r["early_stop"] else " "

        if r["val_bpb"] is not None:
            delta_f = r["val_bpb"] - ref_bpb
            delta_s = f"{delta_f:+.4f}"
        else:
            delta_s = "N/A"

        print(
            f"{tag:<{col_tag}} | {avg_ms:>11} | {steps:>5}{early}| "
            f"{bpb:>7} | {delta_s:>18}"
        )

    print("* = early-stopped by wallclock cap")
    print(SEP)


if __name__ == "__main__":
    main()
