from __future__ import annotations

import hashlib
import json
import pathlib
import re

SUPPORTED_VARIANTS = {"none", "paraphrase", "constraint", "reference", "noisy"}


_BDDL_LANGUAGE_SANITIZE_TABLE = str.maketrans({";": ",", "(": "", ")": ""})


PICK_AND_PLACE_PATTERN = re.compile(
    r"^(?:pick\s+up|pick)\s+(?P<object>.+?)\s+and\s+(?:place|put)\s+it\s+(?P<destination>.+)$",
    re.IGNORECASE,
)
PUT_PATTERN = re.compile(
    r"^put\s+(?P<object>.+?)\s+(?P<destination>(?:on|onto|in|into|to|under|next to|beside|at|on top of|in front of|behind).+)$",
    re.IGNORECASE,
)
STACK_PATTERN = re.compile(r"^stack\s+(?P<object>.+?)\s+on\s+(?P<target>.+)$", re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"^(?P<action>open|close|turn on|turn off)\s+(?P<object>.+?)(?:\s+and\s+(?P<tail>.+))?$",
    re.IGNORECASE,
)


def load_semantic_manifest(manifest_path: str | None) -> dict[str, dict[str, str]]:
    if not manifest_path:
        return {}

    path = pathlib.Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Semantic manifest not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    by_instruction = payload.get("by_instruction", {})
    if not isinstance(by_instruction, dict):
        return {}
    return by_instruction


def rewrite_instruction(instruction: str, variant: str, manifest: dict[str, dict[str, str]] | None = None) -> str:
    normalized_variant = variant.strip().lower()
    if normalized_variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown semantic variant: {variant}. Supported variants: {sorted(SUPPORTED_VARIANTS)}")

    if normalized_variant == "none":
        return instruction

    manifest = manifest or {}
    manifest_item = manifest.get(instruction, {})
    if normalized_variant in manifest_item:
        return _sanitize_bddl_language(manifest_item[normalized_variant])

    return _sanitize_bddl_language(_fallback_rewrite(instruction, normalized_variant))


def _fallback_rewrite(instruction: str, variant: str) -> str:
    text = instruction.strip().rstrip(".")
    lower_text = text.lower()

    if variant == "paraphrase":
        return _paraphrase_instruction(text, lower_text)

    if variant == "constraint":
        return _constraint_instruction(text, lower_text)

    if variant == "reference":
        return _reference_instruction(text, lower_text)

    if variant == "noisy":
        return _noisy_instruction(text)

    return text


def _paraphrase_instruction(text: str, lower_text: str) -> str:
    pick_match = PICK_AND_PLACE_PATTERN.match(lower_text)
    if pick_match:
        obj = pick_match.group("object")
        destination = _replace_spatial_phrase(pick_match.group("destination"), text)
        pick_verb = _det_choice(text, "pick_verb", ["pick up", "grasp", "take"])
        place_verb = _det_choice(text, "place_verb", ["place", "set", "position"])
        templates = [
            f"{pick_verb} {obj} and {place_verb} it {destination}",
            f"{pick_verb} {obj}, then {place_verb} it {destination}",
            f"take hold of {obj} and {place_verb} it {destination}",
        ]
        return _format_instruction(_det_choice(text, "pick_template", templates))

    put_match = PUT_PATTERN.match(lower_text)
    if put_match:
        obj = put_match.group("object")
        destination = _replace_spatial_phrase(put_match.group("destination"), text)
        place_verb = _det_choice(text, "put_place_verb", ["put", "place", "move"])
        templates = [
            f"{place_verb} {obj} {destination}",
            f"carefully {place_verb} {obj} {destination}",
            f"move {obj} so it is {destination}",
        ]
        return _format_instruction(_det_choice(text, "put_template", templates))

    stack_match = STACK_PATTERN.match(lower_text)
    if stack_match:
        obj = stack_match.group("object")
        target = stack_match.group("target")
        templates = [
            f"stack {obj} on top of {target}",
            f"place {obj} over {target} as a stack",
            f"build a stack by putting {obj} on {target}",
        ]
        return _format_instruction(_det_choice(text, "stack_template", templates))

    action_match = ACTION_PATTERN.match(lower_text)
    if action_match:
        action = action_match.group("action")
        obj = action_match.group("object")
        tail = action_match.group("tail")
        action_synonyms = {
            "open": ["open", "pull open", "slide open"],
            "close": ["close", "shut"],
            "turn on": ["turn on", "switch on"],
            "turn off": ["turn off", "switch off"],
        }
        action_text = _det_choice(text, f"action_{action}", action_synonyms.get(action, [action]))
        if tail:
            templates = [
                f"{action_text} {obj} and then {tail}",
                f"{action_text} {obj}, then {tail}",
                f"first {action_text} {obj}, after that {tail}",
            ]
        else:
            templates = [f"{action_text} {obj}", f"please {action_text} {obj}"]
        return _format_instruction(_det_choice(text, "action_template", templates))

    return _format_instruction(_soft_lexical_rewrite(lower_text, text))


def _constraint_instruction(text: str, lower_text: str) -> str:
    obj = _extract_primary_object(lower_text)
    if obj:
        templates = [
            f"{text}, while only interacting with {obj}",
            f"{text}, and keep all other objects where they are",
            f"{text}, without moving anything except {obj}",
        ]
    else:
        templates = [
            f"{text}, and keep the rest of the scene unchanged",
            f"{text}, while avoiding unnecessary contact with other objects",
        ]
    return _format_instruction(_det_choice(text, "constraint_template", templates))


def _reference_instruction(text: str, lower_text: str) -> str:
    obj = _extract_primary_object(lower_text)
    destination = _extract_destination_phrase(lower_text)
    if obj and destination:
        templates = [
            f"In this setup, focus on {obj} and complete: {text}",
            f"Given the current arrangement, use {obj} as reference and execute: {text}",
            f"For the object {obj}, follow this instruction: {text}",
        ]
    elif obj:
        templates = [
            f"In this scene, focus on {obj} and do the following: {text}",
            f"Using {obj} as the reference object, execute: {text}",
        ]
    else:
        templates = [
            f"In the current scene, {text}",
            f"Given this tabletop setup, {text}",
        ]
    return _format_instruction(_det_choice(text, "reference_template", templates))


def _noisy_instruction(text: str) -> str:
    prefix = _det_choice(
        text,
        "noisy_prefix",
        ["Quick note:", "Just to confirm,", "When ready,", "Small reminder,"],
    )
    suffix = _det_choice(
        text,
        "noisy_suffix",
        ["thanks.", "that is the whole task.", "please proceed.", "that's it."],
    )
    templates = [
        f"{prefix} {text}, {suffix}",
        f"{prefix} please {text}, {suffix}",
        f"{prefix} {text}, {suffix}",
    ]
    return _format_instruction(_det_choice(text, "noisy_template", templates))


def _sanitize_bddl_language(text: str) -> str:
    return _format_instruction(text.translate(_BDDL_LANGUAGE_SANITIZE_TABLE))


def _replace_spatial_phrase(destination: str, seed_text: str) -> str:
    spatial_map = {
        "next to": ["next to", "beside"],
        "on top of": ["on top of", "onto"],
        "in front of": ["in front of", "ahead of"],
        "to the right of": ["to the right of", "on the right side of"],
        "to the left of": ["to the left of", "on the left side of"],
    }

    updated = destination
    for source, options in spatial_map.items():
        if source in updated:
            updated = updated.replace(source, _det_choice(seed_text, f"spatial_{source}", options), 1)
    return updated


def _soft_lexical_rewrite(lower_text: str, seed_text: str) -> str:
    rewrites = {
        "pick up": ["pick up", "grasp"],
        "pick ": ["pick ", "grab "],
        "put ": ["put ", "place "],
        "next to": ["next to", "beside"],
    }
    rewritten = lower_text
    for token, candidates in rewrites.items():
        if token in rewritten:
            rewritten = rewritten.replace(token, _det_choice(seed_text, f"rewrite_{token}", candidates), 1)
    return rewritten


def _extract_primary_object(lower_text: str) -> str | None:
    for pattern in [PICK_AND_PLACE_PATTERN, PUT_PATTERN, STACK_PATTERN, ACTION_PATTERN]:
        match = pattern.match(lower_text)
        if not match:
            continue
        if "object" in match.groupdict() and match.group("object"):
            return match.group("object")
    return None


def _extract_destination_phrase(lower_text: str) -> str | None:
    pick_match = PICK_AND_PLACE_PATTERN.match(lower_text)
    if pick_match:
        return pick_match.group("destination")

    put_match = PUT_PATTERN.match(lower_text)
    if put_match:
        return put_match.group("destination")

    return None


def _det_choice(seed_text: str, key: str, options: list[str]) -> str:
    digest = hashlib.sha256(f"{seed_text}|{key}".encode()).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def _format_instruction(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    if not text:
        return text
    return text[0].upper() + text[1:]
