"""
compare.py — side-by-side comparison of Baseline vs DyVM experiment results.

Usage:
    python vim/experiments/compare.py \
        --baseline output/baseline_vim_tiny \
        --dyvm     output/dyvm_vim_tiny_ratio0.7

Each output directory is expected to contain:
  - log.txt        (one JSON line per epoch, written by main.py during training)
  - results.json   (written by main.py --eval --time-measure)

Prints a formatted comparison table to stdout and writes comparison.csv
to the current directory.
"""

import argparse
import json
import csv
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def load_results(output_dir: Path) -> dict:
    """Load results.json from an experiment output directory."""
    path = output_dir / "results.json"
    if not path.exists():
        print(f"[warn] {path} not found — run eval.sh first", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def load_log(output_dir: Path) -> list[dict]:
    """Load all epoch log entries from log.txt."""
    path = output_dir / "log.txt"
    if not path.exists():
        print(f"[warn] {path} not found", file=sys.stderr)
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def best_epoch(log: list[dict]) -> dict | None:
    """Return the log entry with highest test_acc1."""
    if not log:
        return None
    return max(log, key=lambda e: e.get("test_acc1", 0.0))


def final_epoch(log: list[dict]) -> dict | None:
    """Return the last logged epoch."""
    return log[-1] if log else None


def format_val(val, fmt=".3f", missing="—"):
    if val is None:
        return missing
    if isinstance(val, float):
        return format(val, fmt)
    return str(val)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs DyVM experiment results")
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Output directory of the baseline run")
    parser.add_argument("--dyvm", required=True, type=Path,
                        help="Output directory of the DyVM run")
    parser.add_argument("--csv", default="comparison.csv",
                        help="Path to write the CSV summary (default: comparison.csv)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    baseline_results = load_results(args.baseline)
    dyvm_results     = load_results(args.dyvm)

    baseline_log = load_log(args.baseline)
    dyvm_log     = load_log(args.dyvm)

    baseline_best  = best_epoch(baseline_log)
    dyvm_best      = best_epoch(dyvm_log)
    baseline_final = final_epoch(baseline_log)
    dyvm_final     = final_epoch(dyvm_log)

    # ── Build rows ───────────────────────────────────────────────────────────

    def g(d, *keys, default=None):
        """Nested get from dict, returning default if any key is missing."""
        for k in keys:
            if d is None or k not in d:
                return default
            d = d[k]
        return d

    rows = [
        # label, baseline value, dyvm value, unit, lower-is-better?
        # -- Architecture
        ("Parameters",
         g(baseline_results, "n_parameters"),
         g(dyvm_results,     "n_parameters"),
         "params", False),

        # -- Accuracy (from results.json / eval pass)
        ("Top-1 Accuracy (eval)",
         g(baseline_results, "acc1"),
         g(dyvm_results,     "acc1"),
         "%", False),
        ("Top-5 Accuracy (eval)",
         g(baseline_results, "acc5"),
         g(dyvm_results,     "acc5"),
         "%", False),
        ("Top-1 Accuracy EMA (eval)",
         g(baseline_results, "acc1_ema"),
         g(dyvm_results,     "acc1_ema"),
         "%", False),

        # -- Best training epoch
        ("Best Top-1 (training log)",
         g(baseline_best,  "test_acc1"),
         g(dyvm_best,      "test_acc1"),
         "%", False),
        ("Epoch of best acc1",
         g(baseline_best,  "epoch"),
         g(dyvm_best,      "epoch"),
         "epoch", None),
        ("Train loss at best epoch",
         g(baseline_best,  "train_loss"),
         g(dyvm_best,      "train_loss"),
         "", True),

        # -- Final training epoch
        ("Final epoch train loss",
         g(baseline_final, "train_loss"),
         g(dyvm_final,     "train_loss"),
         "", True),
        ("Final epoch test acc1",
         g(baseline_final, "test_acc1"),
         g(dyvm_final,     "test_acc1"),
         "%", False),

        # -- Speed / memory (from results.json after --time-measure)
        ("Latency (ms / batch)",
         g(baseline_results, "latency_sec", default=None),
         g(dyvm_results,     "latency_sec", default=None),
         "ms", True),
        ("Throughput",
         g(baseline_results, "throughput_img_per_sec"),
         g(dyvm_results,     "throughput_img_per_sec"),
         "img/s", False),
        ("Peak GPU memory",
         g(baseline_results, "peak_gpu_mem_mb"),
         g(dyvm_results,     "peak_gpu_mem_mb"),
         "MB", True),

        # -- Token efficiency (DyVM-only; baseline will show —)
        ("Token keep ratio",
         g(baseline_results, "token_keep_ratio"),
         g(dyvm_results,     "token_keep_ratio"),
         "", None),
        ("Token sparsity",
         g(baseline_results, "token_sparsity"),
         g(dyvm_results,     "token_sparsity"),
         "", None),
    ]

    # ── Derived: speedup / accuracy delta ────────────────────────────────────
    def speedup(b_sec, d_sec):
        if b_sec and d_sec:
            return f"{b_sec / d_sec:.2f}x"
        return "—"

    def delta(b, d, scale=1.0):
        if b is not None and d is not None:
            diff = (d - b) * scale
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.2f}"
        return "—"

    # ── Print table ──────────────────────────────────────────────────────────
    col_w = [38, 16, 16, 10]
    header = ["Metric", "Baseline", "DyVM", "Unit"]
    sep = "  ".join("-" * w for w in col_w)

    print()
    print("=" * sum(col_w))
    print("  Baseline vs DyVM — Experiment Comparison")
    print(f"  Baseline : {args.baseline}")
    print(f"  DyVM     : {args.dyvm}")
    print("=" * sum(col_w))
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)

    csv_rows = [header + ["Delta / Note"]]

    for label, b_val, d_val, unit, lower_is_better in rows:
        # Format values
        if label == "Parameters":
            b_str = f"{b_val:,}" if b_val is not None else "—"
            d_str = f"{d_val:,}" if d_val is not None else "—"
        elif label == "Latency (ms / batch)":
            b_str = f"{b_val*1000:.2f}" if b_val is not None else "—"
            d_str = f"{d_val*1000:.2f}" if d_val is not None else "—"
        elif isinstance(b_val, float) or isinstance(d_val, float):
            b_str = format_val(b_val, ".3f")
            d_str = format_val(d_val, ".3f")
        else:
            b_str = format_val(b_val, "d" if isinstance(b_val, int) else ".3f")
            d_str = format_val(d_val, "d" if isinstance(d_val, int) else ".3f")

        # Highlight winner
        note = ""
        if lower_is_better is True and b_val is not None and d_val is not None:
            note = "DyVM faster" if d_val < b_val else ("Baseline faster" if b_val < d_val else "tied")
        elif lower_is_better is False and b_val is not None and d_val is not None:
            note = "DyVM better" if d_val > b_val else ("Baseline better" if b_val > d_val else "tied")

        print("  ".join([
            label.ljust(col_w[0]),
            b_str.ljust(col_w[1]),
            d_str.ljust(col_w[2]),
            unit.ljust(col_w[3]),
        ]))
        csv_rows.append([label, b_str, d_str, unit, note])

    print(sep)

    # ── Summary ──────────────────────────────────────────────────────────────
    b_lat = g(baseline_results, "latency_sec")
    d_lat = g(dyvm_results,     "latency_sec")
    b_acc = g(baseline_results, "acc1") or g(baseline_best, "test_acc1")
    d_acc = g(dyvm_results,     "acc1") or g(dyvm_best,     "test_acc1")

    print()
    print("  Summary")
    print(f"  Latency speedup  : {speedup(b_lat, d_lat)}")
    print(f"  Acc1 delta       : {delta(b_acc, d_acc)} pp")
    if g(dyvm_results, "token_sparsity") is not None:
        sp = g(dyvm_results, "token_sparsity")
        print(f"  Tokens pruned    : {sp*100:.1f}%")
    print()

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"  CSV written to {csv_path}")
    print()


if __name__ == "__main__":
    main()
