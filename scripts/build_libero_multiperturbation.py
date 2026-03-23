"""
Build script for generating multi-dimensional LIBERO perturbations.

Compatibility goals:
- Keep multi-dimensional outputs (semantic/position/object/goal)
- Preserve legacy semantic workflow compatibility:
  - semantic rewrite operates on `(:language ...)` text only
  - Gemini refiner is called with (original, stage1, variant)
  - optional legacy semantic file names: `task__paraphrase.bddl`
  - optional legacy semantic manifest format with `by_instruction`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re

import tyro

from examples.libero.gemini_rewriter import GeminiConfig
from examples.libero.gemini_rewriter import GeminiInstructionRefiner
from examples.libero.goal_rewriter import rewrite_goal_variants
from examples.libero.object_rewriter import rewrite_object_variants
from examples.libero.position_rewriter import rewrite_position_variants
from examples.libero.semantic_utils import rewrite_instruction

LANGUAGE_PATTERN = re.compile(r"\(:language\s+(.+?)\)")
_BDDL_LANGUAGE_SANITIZE_TABLE = str.maketrans({";": ",", "(": "", ")": ""})
RANGE_TUPLE_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)"
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LIBERO_SUITES = {
    "libero_spatial": "Spatial relations (on, in, above, below, left, right)",
    "libero_object": "Object composition (pick between/from/with)",
    "libero_goal": "Goal-conditioned tasks (reach varying targets)",
    "libero_10": "Core 10 tasks (diverse primitives)",
}


def load_bddl_file(bddl_path: Path) -> str:
    with open(bddl_path, encoding="utf-8") as f:
        return f.read()


def save_bddl_file(bddl_path: Path, content: str) -> None:
    bddl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bddl_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_task_name(bddl_path: Path) -> str:
    return bddl_path.stem


def _extract_language_line(content: str) -> tuple[str, tuple[int, int]]:
    match = LANGUAGE_PATTERN.search(content)
    if match is None:
        raise ValueError("No (:language ...) line found in BDDL file")
    return match.group(1).strip(), match.span(1)


def _replace_language_line(content: str, new_language: str) -> str:
    _, (start, end) = _extract_language_line(content)
    safe_language = re.sub(r"\s+", " ", new_language.translate(_BDDL_LANGUAGE_SANITIZE_TABLE)).strip()
    return f"{content[:start]}{safe_language}{content[end:]}"


def _validate_bddl_content(content: str) -> tuple[bool, str | None]:
    """Basic BDDL validation to avoid writing obviously broken files.

    Checks:
    1) Parentheses balance
    2) Every numeric range tuple follows LIBERO ordering: x2 >= x1 and y2 >= y1
    """
    if content.count("(") != content.count(")"):
        return False, "parentheses_mismatch"

    for match in RANGE_TUPLE_PATTERN.finditer(content):
        x1 = float(match.group(1))
        y1 = float(match.group(2))
        x2 = float(match.group(3))
        y2 = float(match.group(4))
        if x2 < x1 or y2 < y1:
            return False, f"invalid_range_order: ({x1} {y1} {x2} {y2})"

    return True, None


def _build_semantic_variants(
    bddl_content: str,
    semantic_variants: list[str],
    llm_refiner: GeminiInstructionRefiner | None,
    llm_enabled: bool,
    llm_variants: set[str],
    llm_status_counts: dict[str, int],
    source_counts: dict[str, int],
    variant_source_counts: dict[str, dict[str, int]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Build semantic variants and return both BDDL variants and legacy manifest info."""
    semantic_bddl_variants = {"none": bddl_content}

    original_instruction, _ = _extract_language_line(bddl_content)
    manifest_item = {
        "original": original_instruction,
        "variants": {},
        "variant_source": {},
    }

    for variant in semantic_variants:
        stage1_rewritten = rewrite_instruction(original_instruction, variant)
        rewritten = stage1_rewritten
        source = "rule"

        if llm_enabled and llm_refiner is not None and variant in llm_variants:
            rewritten, llm_status = llm_refiner.refine(original_instruction, stage1_rewritten, variant)
            llm_status_counts[llm_status] = llm_status_counts.get(llm_status, 0) + 1
            if llm_status == "accepted":
                source = "llm"
            else:
                source = f"rule({llm_status})"

        source_counts[source] = source_counts.get(source, 0) + 1
        variant_source_counts.setdefault(variant, {})
        variant_source_counts[variant][source] = variant_source_counts[variant].get(source, 0) + 1

        semantic_bddl_variants[variant] = _replace_language_line(bddl_content, rewritten)
        manifest_item["variants"][variant] = rewritten
        manifest_item["variant_source"][variant] = source

    return semantic_bddl_variants, manifest_item


def build_multi_perturbation_variants(
    bddl_content: str,
    semantic_variants: list[str],
    llm_refiner: GeminiInstructionRefiner | None,
    llm_enabled: bool,
    llm_variants: set[str],
    perturbation_types: list[str],
    llm_status_counts: dict[str, int],
    source_counts: dict[str, int],
    variant_source_counts: dict[str, dict[str, int]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]] | None]:
    variants: dict[str, dict[str, str]] = {}
    semantic_manifest_item: dict[str, dict[str, str]] | None = None

    if "semantic" in perturbation_types:
        semantic_vars, semantic_manifest_item = _build_semantic_variants(
            bddl_content=bddl_content,
            semantic_variants=semantic_variants,
            llm_refiner=llm_refiner,
            llm_enabled=llm_enabled,
            llm_variants=llm_variants,
            llm_status_counts=llm_status_counts,
            source_counts=source_counts,
            variant_source_counts=variant_source_counts,
        )
        variants["semantic"] = semantic_vars

    if "position" in perturbation_types:
        position_vars = {"none": bddl_content}
        position_vars.update(rewrite_position_variants(bddl_content))
        variants["position"] = position_vars

    if "object" in perturbation_types:
        object_vars = {"none": bddl_content}
        object_vars.update(rewrite_object_variants(bddl_content, num_variants=2))
        variants["object"] = object_vars

    if "goal" in perturbation_types:
        goal_vars = {"none": bddl_content}
        goal_vars.update(rewrite_goal_variants(bddl_content, num_variants=2, llm_refiner=None, llm_enabled=False))
        variants["goal"] = goal_vars

    return variants, semantic_manifest_item


def main(
    libero_root: Path = Path("/home/gaoming/openpi/third_party/libero/libero/libero"),
    output_root: Path = Path("/home/gaoming/openpi/third_party/libero/libero/libero/bddl_files_multiperturbation"),
    manifest_path: Path = Path("/home/gaoming/openpi/libero_data/libero_multiperturbation_variants.json"),
    suites: list[str] | None = None,
    perturbation_types: list[str] | None = None,
    semantic_variants: list[str] | None = None,
    enable_llm_stage: bool = False,  # 如果需要llm介入，要--enable-llm-stage，并且确保gemini_api_key正确配置
    llm_variants: list[str] | None = None,
    gemini_api_key: str | None = None,
    gemini_api_base: str = "https://api2.xcodecli.com/",
    gemini_model: str = "gemini-2.5-flash",
    gemini_temperature: float = 0.2,
    gemini_max_output_tokens: int = 128,
    gemini_timeout_sec: int = 20,
    write_semantic_legacy_filenames: bool = True,
    legacy_semantic_manifest_out: str = "libero_data/libero_semantic_enhanced/instruction_variants.json",
    log_progress_every: int = 5,
    log_verbose: bool = False,
) -> None:
    if suites is None:
        suites = list(LIBERO_SUITES.keys())
    if perturbation_types is None:
        perturbation_types = ["semantic", "position", "object", "goal"]
    if semantic_variants is None:
        semantic_variants = ["paraphrase", "constraint", "reference", "noisy"]
    if llm_variants is None:
        llm_variants = list(semantic_variants)

    if log_verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    llm_refiner = None
    if enable_llm_stage:
        try:
            import os

            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if api_key:
                config = GeminiConfig(
                    api_key=api_key,
                    api_base=gemini_api_base,
                    model=gemini_model,
                    temperature=gemini_temperature,
                    max_output_tokens=gemini_max_output_tokens,
                    timeout_sec=gemini_timeout_sec,
                )
                llm_refiner = GeminiInstructionRefiner(config)
                logger.info("LLM refiner initialized: %s at %s", gemini_model, gemini_api_base)
            else:
                logger.warning("LLM stage enabled but GEMINI_API_KEY not found")
                enable_llm_stage = False
        except Exception as exc:
            logger.error("Failed to initialize LLM refiner: %s", exc)
            enable_llm_stage = False

    llm_variants_set = set(llm_variants)
    llm_status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    variant_source_counts: dict[str, dict[str, int]] = {variant: {} for variant in semantic_variants}

    manifest_data = {
        "metadata": {
            "llm_stage": {
                "enabled": enable_llm_stage,
                "active": bool(enable_llm_stage and llm_refiner is not None),
                "api_base": gemini_api_base,
                "model": gemini_model if enable_llm_stage else None,
                "llm_variants": sorted(llm_variants_set),
                "status_counts": llm_status_counts,
            },
            "perturbation_types": perturbation_types,
            "semantic_variants": semantic_variants,
            "suites": suites,
        },
        "by_suite": {},
        "by_task": {},
        "by_instruction": {},
    }

    total_tasks = 0
    processed_tasks = 0
    failed_tasks: list[tuple[str, str, str]] = []
    generated_files = 0

    logger.info("Building perturbations for suites: %s", suites)
    logger.info("Perturbation types: %s", perturbation_types)

    for suite in suites:
        bddl_dir = libero_root / "bddl_files" / suite
        if not bddl_dir.exists():
            logger.warning("Suite directory not found: %s", bddl_dir)
            continue

        logger.info("\nProcessing suite: %s", suite)
        manifest_data["by_suite"][suite] = {}

        bddl_files = sorted(bddl_dir.glob("*.bddl"))
        suite_total = len(bddl_files)
        total_tasks += suite_total

        for idx, bddl_path in enumerate(bddl_files):
            task_name = extract_task_name(bddl_path)
            task_key = f"{suite}/{task_name}"

            try:
                bddl_content = load_bddl_file(bddl_path)
                all_variants, semantic_manifest_item = build_multi_perturbation_variants(
                    bddl_content=bddl_content,
                    semantic_variants=semantic_variants,
                    llm_refiner=llm_refiner,
                    llm_enabled=enable_llm_stage,
                    llm_variants=llm_variants_set,
                    perturbation_types=perturbation_types,
                    llm_status_counts=llm_status_counts,
                    source_counts=source_counts,
                    variant_source_counts=variant_source_counts,
                )

                suite_output = output_root / suite
                for ptype, ptype_variants in all_variants.items():
                    for var_name, var_bddl in ptype_variants.items():
                        if var_name == "none":
                            continue

                        is_valid, err = _validate_bddl_content(var_bddl)
                        if not is_valid:
                            logger.warning(
                                "Skipping invalid variant %s/%s__%s_%s (%s)",
                                suite,
                                task_name,
                                ptype,
                                var_name,
                                err,
                            )
                            continue

                        # Canonical multiperturbation filename
                        canonical_output = suite_output / f"{task_name}__{ptype}_{var_name}.bddl"
                        save_bddl_file(canonical_output, var_bddl)
                        generated_files += 1

                        # Legacy semantic filename for backward compatibility
                        if ptype == "semantic" and write_semantic_legacy_filenames:
                            legacy_output = suite_output / f"{task_name}__{var_name}.bddl"
                            save_bddl_file(legacy_output, var_bddl)
                            generated_files += 1

                manifest_data["by_suite"][suite][task_name] = {
                    "perturbation_counts": {
                        ptype: max(0, len(ptype_vars) - 1) for ptype, ptype_vars in all_variants.items()
                    },
                    "total_variants": sum(max(0, len(ptype_vars) - 1) for ptype_vars in all_variants.values()),
                }

                if semantic_manifest_item is not None:
                    manifest_data["by_task"][task_key] = {
                        "original": semantic_manifest_item["original"],
                        "variants": semantic_manifest_item["variants"],
                        "variant_source": semantic_manifest_item["variant_source"],
                    }
                    manifest_data["by_instruction"][semantic_manifest_item["original"]] = semantic_manifest_item[
                        "variants"
                    ]

                processed_tasks += 1
                if (idx + 1) % log_progress_every == 0:
                    logger.info("  %s: Processed %d/%d tasks", suite, idx + 1, suite_total)

            except Exception as exc:
                logger.error("Failed to process %s: %s", task_name, exc)
                failed_tasks.append((suite, task_name, str(exc)))

        logger.info("  %s: Completed %d tasks", suite, len(bddl_files))

    manifest_data["metadata"]["generated_files"] = generated_files
    manifest_data["metadata"]["source_counts"] = source_counts
    manifest_data["metadata"]["variant_source_counts"] = variant_source_counts

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    # Optional legacy semantic manifest path for drop-in compatibility.
    if "semantic" in perturbation_types and legacy_semantic_manifest_out:
        legacy_path = Path(legacy_semantic_manifest_out)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "metadata": {
                "suites": suites,
                "variants": semantic_variants,
                "llm_stage": manifest_data["metadata"]["llm_stage"],
                "variant_source_counts": variant_source_counts,
                "source_counts": source_counts,
            },
            "by_task": manifest_data["by_task"],
            "by_instruction": manifest_data["by_instruction"],
        }
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f, ensure_ascii=False, indent=2)
        logger.info("Legacy semantic manifest saved to: %s", legacy_path)

    logger.info("\n%s", "=" * 60)
    logger.info("Perturbation generation complete!")
    logger.info("Total tasks: %d", total_tasks)
    logger.info("Successfully processed: %d", processed_tasks)
    logger.info("Failed: %d", len(failed_tasks))
    logger.info("Generated files: %d", generated_files)
    logger.info("Output directory: %s", output_root)
    logger.info("Manifest saved to: %s", manifest_path)

    if enable_llm_stage:
        total_llm_calls = sum(llm_status_counts.values())
        accepted = llm_status_counts.get("accepted", 0)
        accept_rate = 0.0 if total_llm_calls == 0 else accepted / total_llm_calls * 100.0
        logger.info("LLM accepted: %d/%d (%.1f%%)", accepted, total_llm_calls, accept_rate)

    if failed_tasks:
        logger.warning("\nFailed tasks:")
        for suite, task, error in failed_tasks:
            logger.warning("  %s/%s: %s", suite, task, error)


if __name__ == "__main__":
    tyro.cli(main)
