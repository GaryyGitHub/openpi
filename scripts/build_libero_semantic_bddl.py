from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import tyro

from examples.libero.gemini_rewriter import GeminiConfig
from examples.libero.gemini_rewriter import GeminiInstructionRefiner
from examples.libero.semantic_utils import SUPPORTED_VARIANTS
from examples.libero.semantic_utils import rewrite_instruction

LANGUAGE_PATTERN = re.compile(r"\(:language\s+(.+?)\)")
_BDDL_LANGUAGE_SANITIZE_TABLE = str.maketrans({";": ",", "(": "", ")": ""})


@dataclasses.dataclass
class Args:
    bddl_root: str = "third_party/libero/libero/libero/bddl_files"
    output_root: str = "third_party/libero/libero/libero/bddl_files_semantic_nollm"
    manifest_out: str = "libero_data/libero_semantic_enhanced/instruction_variants_nollm.json"
    suites: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    variants: tuple[str, ...] = ("paraphrase", "constraint", "reference", "noisy")
    enable_llm_stage: bool = False
    llm_variants: tuple[str, ...] = ("paraphrase", "constraint", "reference", "noisy")
    gemini_api_key: str = ""
    gemini_api_base: str = "https://api2.xcodecli.com/"
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 128
    gemini_timeout_sec: int = 20
    log_progress_every: int = 5
    log_llm_samples: int = 8
    log_verbose: bool = True


def _extract_language_line(content: str) -> tuple[str, tuple[int, int]]:
    match = LANGUAGE_PATTERN.search(content)
    if match is None:
        raise ValueError("No (:language ...) line found in BDDL file")
    return match.group(1).strip(), match.span(1)


def _replace_language_line(content: str, new_language: str) -> str:
    _, (start, end) = _extract_language_line(content)
    safe_language = re.sub(r"\s+", " ", new_language.translate(_BDDL_LANGUAGE_SANITIZE_TABLE)).strip()
    return f"{content[:start]}{safe_language}{content[end:]}"


def main(args: Args) -> None:
    invalid = [variant for variant in args.variants if variant not in SUPPORTED_VARIANTS or variant == "none"]
    if invalid:
        raise ValueError(f"Invalid variants: {invalid}. Use subset of {sorted(SUPPORTED_VARIANTS - {'none'})}.")

    bddl_root = pathlib.Path(args.bddl_root)
    output_root = pathlib.Path(args.output_root)
    manifest_out = pathlib.Path(args.manifest_out)
    llm_variants = {variant for variant in args.llm_variants if variant in set(args.variants)}

    llm_refiner = GeminiInstructionRefiner(
        GeminiConfig(
            api_key=args.gemini_api_key,
            api_base=args.gemini_api_base,
            model=args.gemini_model,
            temperature=args.gemini_temperature,
            max_output_tokens=args.gemini_max_output_tokens,
            timeout_sec=args.gemini_timeout_sec,
        )
    )
    use_llm_stage = args.enable_llm_stage and llm_refiner.enabled
    llm_status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    variant_source_counts: dict[str, dict[str, int]] = {variant: {} for variant in args.variants}
    llm_samples: list[dict[str, str]] = []

    by_task: dict[str, dict] = {}
    by_instruction: dict[str, dict[str, str]] = {}
    generated_files = 0
    task_counter = 0

    if args.log_verbose:
        print("=== LIBERO Semantic Build ===")
        print(f"Suites: {args.suites}")
        print(f"Variants: {args.variants}")
        print(f"LLM stage requested: {args.enable_llm_stage}")
        print(f"LLM stage active: {use_llm_stage}")
        if args.enable_llm_stage:
            print(f"LLM endpoint: {args.gemini_api_base}")
            print(f"LLM model: {args.gemini_model}")

    for suite in args.suites:
        input_suite_dir = bddl_root / suite
        if not input_suite_dir.exists():
            continue

        if args.log_verbose:
            print(f"\n[Suite] {suite}")

        output_suite_dir = output_root / suite
        output_suite_dir.mkdir(parents=True, exist_ok=True)

        suite_files = sorted(input_suite_dir.glob("*.bddl"))
        for bddl_file in suite_files:
            original_content = bddl_file.read_text(encoding="utf-8")
            original_instruction, _ = _extract_language_line(original_content)
            task_counter += 1

            task_key = f"{suite}/{bddl_file.stem}"
            task_item = {
                "original": original_instruction,
                "variants": {},
                "variant_source": {},
                "generated_files": {},
            }

            by_instruction.setdefault(original_instruction, {})

            for variant in args.variants:
                stage1_rewritten = rewrite_instruction(original_instruction, variant)
                rewritten = stage1_rewritten
                source = "rule"
                llm_status = "not_used"
                if use_llm_stage and variant in llm_variants:
                    rewritten, llm_status = llm_refiner.refine(original_instruction, rewritten, variant)
                    llm_status_counts[llm_status] = llm_status_counts.get(llm_status, 0) + 1
                    if llm_status == "accepted":
                        source = "llm"
                    else:
                        source = f"rule({llm_status})"

                    if len(llm_samples) < args.log_llm_samples:
                        llm_samples.append(
                            {
                                "task": task_key,
                                "variant": variant,
                                "status": llm_status,
                                "original": original_instruction,
                                "stage1": stage1_rewritten,
                                "final": rewritten,
                            }
                        )

                source_counts[source] = source_counts.get(source, 0) + 1
                variant_source_counts[variant][source] = variant_source_counts[variant].get(source, 0) + 1
                target_name = f"{bddl_file.stem}__{variant}.bddl"
                target_path = output_suite_dir / target_name

                target_content = _replace_language_line(original_content, rewritten)
                target_path.write_text(target_content, encoding="utf-8")

                task_item["variants"][variant] = rewritten
                task_item["variant_source"][variant] = source
                task_item["generated_files"][variant] = str(target_path.as_posix())
                by_instruction[original_instruction][variant] = rewritten
                generated_files += 1

            by_task[task_key] = task_item

            if args.log_verbose and args.log_progress_every > 0 and task_counter % args.log_progress_every == 0:
                print(
                    f"Progress: {task_counter} tasks processed, {generated_files} variant files generated "
                    f"(latest task: {task_key})"
                )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "bddl_root": str(bddl_root.as_posix()),
            "output_root": str(output_root.as_posix()),
            "suites": list(args.suites),
            "variants": list(args.variants),
            "llm_stage": {
                "enabled": args.enable_llm_stage,
                "active": use_llm_stage,
                "api_base": args.gemini_api_base,
                "model": args.gemini_model,
                "llm_variants": sorted(llm_variants),
                "status_counts": llm_status_counts,
            },
            "variant_source_counts": variant_source_counts,
            "source_counts": source_counts,
            "generated_files": generated_files,
        },
        "by_task": by_task,
        "by_instruction": by_instruction,
    }
    manifest_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {generated_files} semantic BDDL files.")
    if args.enable_llm_stage and not llm_refiner.enabled:
        print("LLM stage requested but no Gemini API key found. Using rule-only outputs.")
    if use_llm_stage:
        print(f"LLM stage status counts: {llm_status_counts}")
        total_llm_calls = sum(llm_status_counts.values())
        accepted = llm_status_counts.get("accepted", 0)
        accept_rate = 0.0 if total_llm_calls == 0 else accepted / total_llm_calls * 100.0
        print(f"LLM accepted: {accepted}/{total_llm_calls} ({accept_rate:.1f}%)")

    print(f"Source counts: {source_counts}")
    print(f"Per-variant source counts: {variant_source_counts}")

    if llm_samples:
        print("\nLLM sample previews (rule vs final):")
        for idx, sample in enumerate(llm_samples, start=1):
            print(
                f"[{idx}] {sample['task']} | {sample['variant']} | {sample['status']}\n"
                f"  original: {sample['original']}\n"
                f"  stage1  : {sample['stage1']}\n"
                f"  final   : {sample['final']}"
            )

    print(f"Manifest written to: {manifest_out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
