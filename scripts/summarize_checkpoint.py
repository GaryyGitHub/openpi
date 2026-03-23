#!/usr/bin/env python
"""
用法：python scripts/summarize_checkpoint.py /data/gaoming/openpi/checkpoints/pi05_libero/libero_lora_2/29999/params --top 8 --csv /data/gaoming/openpi/checkpoints/pi05_libero/libero_lora_2/29999
输入为array_metadatas，输出参数统计信息，路径为checkpoint下的params目录
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def product(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


def collect_array_metadatas(params_dir: Path):
    """Yield (param_name, write_shape) across all process_* files under params_dir.

    Expects files under params_dir/**/params/array_metadatas/process_*
    """
    meta_dir = params_dir / "array_metadatas"
    if not meta_dir.exists():
        # Try to find recursively
        for p in params_dir.rglob("array_metadatas/process_*"):
            yield from _read_process_file(p)
        return
    for proc_file in sorted(meta_dir.glob("process_*")):
        yield from _read_process_file(proc_file)


def _read_process_file(proc_file: Path):
    try:
        with open(proc_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {proc_file}: {e}")
        return []
    arrs = data.get("array_metadatas", [])
    for item in arrs:
        md = item.get("array_metadata", {})
        name = md.get("param_name")
        shape = md.get("write_shape") or md.get("chunk_shape") or []
        if name and shape:
            yield name, tuple(int(s) for s in shape)


def main():
    ap = argparse.ArgumentParser(description="Summarize parameter counts from OpenPI checkpoint metadatas")
    ap.add_argument(
        "checkpoint_params_dir", type=Path, help="Path to the params directory (e.g., checkpoints/.../params)"
    )
    ap.add_argument("--top", type=int, default=10, help="Show top-N largest arrays")
    ap.add_argument(
        "--csv", type=Path, default=None, help="Directory to write CSV outputs (params_breakdown.csv, summary.csv)"
    )
    args = ap.parse_args()

    # Aggregate element counts per param across processes (sum shards)
    per_param_counts = defaultdict(int)
    per_param_shape_examples = {}
    for name, shape in collect_array_metadatas(args.checkpoint_params_dir):
        per_param_counts[name] += product(shape)
        per_param_shape_examples.setdefault(name, shape)

    total = sum(per_param_counts.values())

    def is_lora(n):
        return (".lora_a" in n) or (".lora_b" in n)

    def is_llm(n):
        return ".llm." in n

    def is_llm_expert(n):
        # Matches action expert as used by freeze filter: ".*llm.*_1.*"
        return ".llm." in n and "_1" in n

    def is_llm_main(n):
        return ".llm." in n and "_1" not in n

    def is_img(n):
        return ".img." in n

    def is_action_proj(n):
        return (
            any(k in n for k in ["params.action_in_proj", "params.action_out_proj", "params.state_proj"])
            or any(k in n for k in ["params.time_mlp_in", "params.time_mlp_out"])
            or n.startswith("params.action_")
        )

    lora = sum(c for n, c in per_param_counts.items() if is_lora(n))
    non_lora = total - lora
    llm_total = sum(c for n, c in per_param_counts.items() if is_llm(n))
    llm_main = sum(c for n, c in per_param_counts.items() if is_llm_main(n))
    llm_expert = sum(c for n, c in per_param_counts.items() if is_llm_expert(n))
    img_total = sum(c for n, c in per_param_counts.items() if is_img(n))
    proj_total = sum(c for n, c in per_param_counts.items() if is_action_proj(n))

    def pct(x):
        return 0.0 if total == 0 else (100.0 * x / total)

    print("=== Parameter Summary ===")
    print(f"Total parameters (elements): {total:,}")
    print(f"  - LoRA params:        {lora:,} ({pct(lora):.2f}%)")
    print(f"  - Non-LoRA params:    {non_lora:,} ({pct(non_lora):.2f}%)")
    print(f"LLM total:              {llm_total:,} ({pct(llm_total):.2f}%)")
    print(f"  - Paligemma (main):   {llm_main:,} ({pct(llm_main):.2f}%)")
    print(f"  - Action expert:      {llm_expert:,} ({pct(llm_expert):.2f}%)")
    print(f"Vision tower (img):     {img_total:,} ({pct(img_total):.2f}%)")
    print(f"Head/projections:       {proj_total:,} ({pct(proj_total):.2f}%)")

    # Top-N largest arrays
    print("\n=== Top arrays (by elements) ===")
    top = sorted(per_param_counts.items(), key=lambda kv: kv[1], reverse=True)[: args.top]
    for n, c in top:
        print(f"{c:>15,}  {per_param_shape_examples[n]}  {n}")

    # A few sanity cues we can infer
    has_time_mlp = any(n.startswith("params.time_mlp_") for n in per_param_counts)
    print("\n=== Detected features ===")
    print(f"pi05 (adaRMS time MLP): {'yes' if has_time_mlp else 'no'}")
    # Check LoRA ranks by finding typical lora_a shapes last dim
    lora_ranks = defaultdict(int)
    for n, shape in per_param_shape_examples.items():
        if ".lora_a" in n and len(shape) > 0:
            r = shape[-1]
            lora_ranks[r] += 1
    if lora_ranks:
        ranks_str = ", ".join(f"rank={r} (count={c})" for r, c in sorted(lora_ranks.items()))
    else:
        ranks_str = "none"
    print(f"LoRA ranks observed: {ranks_str}")

    # Optional CSV outputs
    if args.csv is not None:
        out_dir = args.csv
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-parameter breakdown
        params_csv = out_dir / "params_breakdown.csv"
        with open(params_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "param_name",
                    "elements",
                    "shape",
                    "is_lora",
                    "module",
                ]
            )
            for n, c in sorted(per_param_counts.items(), key=lambda kv: kv[0]):
                if is_llm_main(n):
                    module = "llm_main"
                elif is_llm_expert(n):
                    module = "llm_expert"
                elif is_img(n):
                    module = "img"
                elif is_action_proj(n):
                    module = "proj"
                else:
                    module = "other"
                writer.writerow(
                    [
                        n,
                        c,
                        "x".join(str(x) for x in per_param_shape_examples[n]),
                        1 if is_lora(n) else 0,
                        module,
                    ]
                )

        # Summary CSV
        summary_csv = out_dir / "summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "elements", "percent"])
            writer.writerow(["total", total, 100.0])
            writer.writerow(["lora", lora, pct(lora)])
            writer.writerow(["non_lora", non_lora, pct(non_lora)])
            writer.writerow(["llm_total", llm_total, pct(llm_total)])
            writer.writerow(["llm_main", llm_main, pct(llm_main)])
            writer.writerow(["llm_expert", llm_expert, pct(llm_expert)])
            writer.writerow(["img_total", img_total, pct(img_total)])
            writer.writerow(["proj_total", proj_total, pct(proj_total)])
            writer.writerow(["pi05_time_mlp", int(has_time_mlp), ""])
            writer.writerow(["lora_ranks", "; ".join(f"{r}:{c}" for r, c in sorted(lora_ranks.items())), ""])
        print(f"\nCSV written to: {params_csv} and {summary_csv}")


if __name__ == "__main__":
    main()
