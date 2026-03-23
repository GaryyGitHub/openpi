"""
Goal/Task Perturbation Rewriter for LIBERO BDDL tasks.

Modifies the `:goal` field by:
- Adding soft constraints (ordering, intermediate steps)
- Changing target location/support surface
- Adding conditional goals (if-then constraints)

Optionally uses Gemini API for semantic validation.

Reference: LIBERO-Pro goal constraint perturbation design.
"""

import hashlib
import re


def _det_choice(seed_text: str, key: str, options: list) -> any:
    """Deterministically pick one option using SHA256 hash."""
    hash_str = hashlib.sha256(f"{seed_text}_{key}".encode()).hexdigest()
    choice_idx = int(hash_str, 16) % len(options)
    return options[choice_idx]


def _parse_goal_section(bddl_content: str) -> tuple[int, int, str] | None:
    """Extract goal section from BDDL."""
    lines = bddl_content.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "(:goal" in line:
            start_idx = i
        elif start_idx is not None and line.strip() == ")":
            goal_content = "\n".join(lines[start_idx : i + 1])
            if goal_content.count("(") == goal_content.count(")"):
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        goal_text = "\n".join(lines[start_idx : end_idx + 1])
        return (start_idx, end_idx, goal_text)

    return None


def _parse_init_section(bddl_content: str) -> tuple[int, int, str] | None:
    """Extract init section from BDDL."""
    lines = bddl_content.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "(:init" in line:
            start_idx = i
        elif start_idx is not None and line.strip() == ")":
            init_content = "\n".join(lines[start_idx : i + 1])
            if init_content.count("(") == init_content.count(")"):
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        init_text = "\n".join(lines[start_idx : end_idx + 1])
        return (start_idx, end_idx, init_text)

    return None


def _extract_goal_constraints(goal_text: str) -> list[str]:
    """Extract individual goal constraints.

    E.g., "(And (On obj1 obj2) (Holding obj3))" -> ["(On obj1 obj2)", "(Holding obj3)"]
    """
    # Remove outer (And ...) if present
    inner = re.sub(r"^\s*\(:goal\s*\(And\s*", "", goal_text)
    inner = re.sub(r"\)\s*\)\s*$", "", inner)

    # Extract individual constraint terms
    constraints = []
    paren_count = 0
    current = []

    for char in inner:
        if char == "(":
            current.append(char)
            paren_count += 1
        elif char == ")":
            current.append(char)
            paren_count -= 1
            if paren_count == 0:
                constraints.append("".join(current).strip())
                current = []
        else:
            current.append(char)

    return [c for c in constraints if c]


def _extract_obj_of_interest_list(bddl_content: str) -> list[str]:
    """Extract ordered obj_of_interest entries from BDDL."""
    match = re.search(r"\(:obj_of_interest\s+(.+?)\)", bddl_content, re.DOTALL)
    if not match:
        return []
    return re.findall(r"\b([a-z_][a-z0-9_]*(?:_\d+)?)\b", match.group(1).lower())


def _extract_declared_entities(bddl_content: str) -> set:
    """Extract declared object / fixture instance names from BDDL."""
    entities = set()

    objects_match = re.search(r"\(:objects\s+(.+?)\)\s*\(:obj_of_interest", bddl_content, re.DOTALL)
    if objects_match:
        entities.update(re.findall(r"\b([a-z_][a-z0-9_]*_\d+)\b", objects_match.group(1).lower()))

    fixtures_match = re.search(r"\(:fixtures\s+(.+?)\)\s*\(:objects", bddl_content, re.DOTALL)
    if fixtures_match:
        entities.update(
            re.findall(r"\b([a-z_][a-z0-9_]*(?:_\d+)?)\s*-\s*[a-z_][a-z0-9_]*\b", fixtures_match.group(1).lower())
        )

    return entities


def _target_to_language_phrase(target: str) -> str:
    """Convert a symbolic target token into a natural-language phrase."""
    if target.endswith("_region"):
        stem = target[:-7]
        if stem.startswith("main_table_"):
            stem = stem[len("main_table_") :]
            words = stem.replace("_", " ")
            return f"the {words} region on the table"
        words = stem.replace("_", " ")
        return f"the {words} region"

    token = re.sub(r"_\d+$", "", target)
    return f"the {token.replace('_', ' ')}"


def _update_language_target(bddl_content: str, new_target: str) -> str:
    """Update :language text so prompt aligns with changed goal target."""
    phrase = _target_to_language_phrase(new_target)

    # Prefer replacing the final placement clause if present.
    updated = re.sub(
        r"(?i)(place\s+it\s+on\s+)(the\s+[^\n\)]+)",
        rf"\1{phrase}",
        bddl_content,
        count=1,
    )
    if updated != bddl_content:
        return updated

    # Fallback: if pattern not found, keep content unchanged.
    return bddl_content


def _update_obj_of_interest_target(bddl_content: str, main_obj: str, new_target: str) -> str:
    """Keep :obj_of_interest consistent with changed target when possible."""
    pattern = re.compile(r"\(:obj_of_interest\s+(.+?)\)", re.DOTALL)
    match = pattern.search(bddl_content)
    if not match:
        return bddl_content

    if new_target.endswith("_region") or new_target == "main_table":
        new_block = f"(:obj_of_interest\n    {main_obj}\n  )"
        return bddl_content[: match.start()] + new_block + bddl_content[match.end() :]

    declared = _extract_declared_entities(bddl_content)
    if new_target not in declared:
        return bddl_content

    new_block = f"(:obj_of_interest\n    {main_obj}\n    {new_target}\n  )"
    return bddl_content[: match.start()] + new_block + bddl_content[match.end() :]


def _entity_to_language_phrase(entity: str) -> str:
    """Convert an object / region token into a readable phrase."""
    if entity.endswith("_region"):
        return _target_to_language_phrase(entity)
    token = re.sub(r"_\d+$", "", entity)
    return f"the {token.replace('_', ' ')}"


def _append_language_constraint(bddl_content: str, obj_name: str, support: str) -> str:
    """Append an explicit extra-goal clause to :language for add_constraint."""
    match = re.search(r"\(:language\s+([^\n\)]+)\)", bddl_content)
    if not match:
        return bddl_content

    original = match.group(1).strip()
    addition = f" Also ensure {_entity_to_language_phrase(obj_name)} is on {_entity_to_language_phrase(support)}"
    if addition.strip() in original:
        return bddl_content

    new_language = original.rstrip(".") + "." + addition + "."
    return bddl_content[: match.start(1)] + new_language + bddl_content[match.end(1) :]


def _update_obj_of_interest_for_constraint(bddl_content: str, obj_name: str, support: str) -> str:
    """Include extra-goal entities in :obj_of_interest when they are declared objects/fixtures."""
    pattern = re.compile(r"\(:obj_of_interest\s+(.+?)\)", re.DOTALL)
    match = pattern.search(bddl_content)
    if not match:
        return bddl_content

    current = _extract_obj_of_interest_list(bddl_content)
    declared = _extract_declared_entities(bddl_content)

    updated: list[str] = []
    for name in current:
        if name not in updated:
            updated.append(name)
    if obj_name in declared and obj_name not in updated:
        updated.append(obj_name)
    if support in declared and support not in updated:
        updated.append(support)

    if not updated:
        return bddl_content

    lines = ["(:obj_of_interest"] + [f"    {name}" for name in updated] + ["  )"]
    new_block = "\n".join(lines)
    return bddl_content[: match.start()] + new_block + bddl_content[match.end() :]


def _constraint_to_language_phrase(constraint: str) -> str | None:
    """Convert a single goal constraint to a short natural-language phrase."""
    match = re.match(r"^\((On|In|Open|Close|Turnon|Turnoff)\s+([^\s\)]+)(?:\s+([^\s\)]+))?\)$", constraint.strip())
    if not match:
        return None

    pred, arg1, arg2 = match.groups()
    if pred == "On" and arg2 is not None:
        return f"put {_entity_to_language_phrase(arg1)} on {_entity_to_language_phrase(arg2)}"
    if pred == "In" and arg2 is not None:
        return f"put {_entity_to_language_phrase(arg1)} in {_entity_to_language_phrase(arg2)}"
    if pred == "Open":
        return f"open {_entity_to_language_phrase(arg1)}"
    if pred == "Close":
        return f"close {_entity_to_language_phrase(arg1)}"
    if pred == "Turnon":
        return f"turn on {_entity_to_language_phrase(arg1)}"
    if pred == "Turnoff":
        return f"turn off {_entity_to_language_phrase(arg1)}"
    return None


def _set_language_change_target(bddl_content: str, new_constraint: str) -> str:
    """Set :language to a concise prompt aligned with the new target constraint."""
    match = re.search(r"\(:language\s+([^\n\)]+)\)", bddl_content)
    if not match:
        return bddl_content

    new_phrase = _constraint_to_language_phrase(new_constraint)
    if new_phrase is None:
        return bddl_content

    new_language = new_phrase[0].upper() + new_phrase[1:].rstrip(".") + "."
    return bddl_content[: match.start(1)] + new_language + bddl_content[match.end(1) :]


def _sync_obj_of_interest_with_goal(bddl_content: str) -> str:
    """Ensure :obj_of_interest includes all declared entities referenced by :goal."""
    pattern = re.compile(r"\(:obj_of_interest\s+(.+?)\)", re.DOTALL)
    match = pattern.search(bddl_content)
    if not match:
        return bddl_content

    goal_parse = _parse_goal_section(bddl_content)
    if goal_parse is None:
        return bddl_content

    _, _, goal_text = goal_parse
    constraints = _extract_goal_constraints(goal_text)
    declared = _extract_declared_entities(bddl_content)

    goal_entities: list[str] = []

    def _region_owner_entity(token: str) -> str | None:
        # Example: basket_1_contain_region -> basket_1
        match = re.match(r"^([a-z_][a-z0-9_]*_\d+)_.+_region$", token)
        return match.group(1) if match else None

    for c in constraints:
        parsed = re.match(r"^\((On|In|Open|Close|Turnon|Turnoff)\s+([^\s\)]+)(?:\s+([^\s\)]+))?\)$", c.strip())
        if parsed:
            _, arg1, arg2 = parsed.groups()
            for token in (arg1, arg2):
                if token is None:
                    continue
                if token in declared and token not in goal_entities:
                    goal_entities.append(token)
                owner = _region_owner_entity(token)
                if owner is not None and owner in declared and owner not in goal_entities:
                    goal_entities.append(owner)
        for token in _extract_objects_from_constraint(c):
            if token in declared and token not in goal_entities:
                goal_entities.append(token)

    updated: list[str] = []
    for name in goal_entities:
        if name not in updated:
            updated.append(name)

    if not updated:
        return bddl_content

    lines = ["(:obj_of_interest"] + [f"    {name}" for name in updated] + ["  )"]
    new_block = "\n".join(lines)
    return bddl_content[: match.start()] + new_block + bddl_content[match.end() :]


def _extract_objects_from_constraint(constraint: str) -> list[str]:
    """Extract object names from a constraint.

    E.g., "(On akita_black_bowl_1 plate_1)" -> ["akita_black_bowl_1", "plate_1"]
    """
    # Extract tokens that look like object names (contain underscores and/or numbers)
    tokens = re.findall(r"\b([a-z_][a-z0-9_]*)\b", constraint.lower())
    # Filter out relation names
    relations = {"on", "holding", "not", "and", "or", "inside", "above", "clear", "in"}
    return [t for t in tokens if t not in relations and "_" in t]


def _get_valid_regions_for_object(obj_name: str, all_regions: list[str]) -> list[str]:
    """Filter regions based on object semantic appropriateness.

    E.g., 'plate_1', 'fork_1', 'spoon_1' (kitchenware) should not go on stove/fridge.
    Objects go on plates/boxes or general table areas, not specialized appliance regions.

    Args:
        obj_name: Object identifier (e.g., 'plate_1', 'spoon_1', 'akita_black_bowl_1')
        all_regions: All available table regions

    Returns:
        Filtered list of semantically valid regions for this object
    """
    # Check if object is kitchenware (plate, fork, spoon, bowl, cup, dish, pot, pan, moka, etc.)
    kitchenware_keywords = {
        "plate",
        "fork",
        "spoon",
        "bowl",
        "cup",
        "dish",
        "knife",
        "glass",
        "utensil",
        "pot",
        "pan",
        "moka",
        "sauce",
        "dressing",
        "cheese",
        "butter",
        "soup",
        "ingredient",
    }
    obj_stem = re.sub(r"_\d+$", "", obj_name).lower()

    is_kitchenware = any(kw in obj_stem for kw in kitchenware_keywords)

    if is_kitchenware:
        # Kitchenware should avoid specialized appliance regions (stove, fridge, sink, etc.) and their sub-regions
        exclude_keywords = {
            "stove",
            "fridge",
            "sink",
            "oven",
            "microwave",
            "dishwasher",
            "_cook_region",
            "_prep_region",
        }
        valid = [r for r in all_regions if not any(ex in r.lower() for ex in exclude_keywords)]
        # If no regions passed filter, return all (fallback to safety)
        return valid if valid else all_regions

    # Non-kitchenware objects can go anywhere
    return all_regions


def _add_ordering_constraint(bddl_content: str, det_key: str = "goal_extra_constraint") -> str:
    """Add an extra goal conjunct by preserving one distractor relation from init.

    This stays within the existing BDDL predicate vocabulary and makes the task
    slightly stricter without changing the main manipulation target.
    """
    goal_parse = _parse_goal_section(bddl_content)
    init_parse = _parse_init_section(bddl_content)
    if goal_parse is None or init_parse is None:
        return bddl_content

    # Extract fixture names to avoid creating impossible constraints
    fixture_names: set = set()
    fixtures_match = re.search(r"\(:fixtures\s+(.+?)\)\s*\(:objects", bddl_content, re.DOTALL)
    if fixtures_match:
        # Extract fixture instance names (e.g., "flat_stove_1" from "flat_stove_1 - flat_stove")
        fixture_names.update(
            re.findall(r"\b([a-z_][a-z0-9_]*_\d+)\s*-\s*[a-z_][a-z0-9_]*\b", fixtures_match.group(1).lower())
        )

    start_idx, end_idx, goal_text = goal_parse
    _, _, init_text = init_parse

    constraints = _extract_goal_constraints(goal_text)
    goal_objects = set()
    for constraint in constraints:
        goal_objects.update(_extract_objects_from_constraint(constraint))

    obj_of_interest = _extract_obj_of_interest_list(bddl_content)
    main_obj = obj_of_interest[0] if obj_of_interest else None
    secondary_interest = set(obj_of_interest[1:])

    # Parse init On-relations.
    init_pairs: list[tuple[str, str]] = []
    for line in init_text.splitlines():
        match = re.search(r"\(On\s+(\S+)\s+([^\s)]+)\)", line.strip())
        if match:
            init_pairs.append(match.groups())

    init_constraints = {f"(On {obj_name} {support})" for obj_name, support in init_pairs}
    init_support_by_obj = dict(init_pairs)

    # Candidate support pool is derived from existing stable table regions.
    region_supports = sorted({support for _, support in init_pairs if support.endswith("_region")})

    preferred_constraints: list[str] = []
    candidate_constraints: list[str] = []

    # Preferred: move secondary task-relevant objects to a new (different) region.
    for obj_name in secondary_interest:
        if obj_name not in init_support_by_obj or obj_name in fixture_names:
            continue
        current_support = init_support_by_obj[obj_name]
        # Filter regions based on object semantic appropriateness
        valid_regions = _get_valid_regions_for_object(obj_name, region_supports)
        for alt_support in valid_regions:
            if alt_support == current_support:
                continue
            candidate = f"(On {obj_name} {alt_support})"
            if candidate in constraints or candidate in init_constraints:
                continue
            preferred_constraints.append(candidate)

    # Fallback: create a new region target for non-goal distractors.
    for obj_name, support in init_pairs:
        if obj_name in goal_objects or support in goal_objects:
            continue
        if main_obj is not None and obj_name == main_obj:
            continue
        if obj_name in fixture_names:
            # Skip fixtures - they cannot be moved
            continue
        # Filter regions based on object semantic appropriateness
        valid_regions = _get_valid_regions_for_object(obj_name, region_supports)
        for alt_support in valid_regions:
            if alt_support == support:
                continue
            candidate = f"(On {obj_name} {alt_support})"
            if candidate in constraints or candidate in init_constraints:
                continue
            candidate_constraints.append(candidate)

    effective_candidates = preferred_constraints if preferred_constraints else candidate_constraints

    if not effective_candidates:
        return bddl_content

    extra_constraint = _det_choice(bddl_content, det_key, effective_candidates)
    if extra_constraint in constraints:
        return bddl_content

    if "(And " in goal_text:
        new_goal = goal_text.replace("(And ", f"(And {extra_constraint} ", 1)
    else:
        inner_match = re.search(r"\(:goal\s*(\(.+\))\s*\)", goal_text, re.DOTALL)
        if not inner_match:
            return bddl_content
        original_constraint = inner_match.group(1).strip()
        new_goal = goal_text.replace(original_constraint, f"(And {original_constraint} {extra_constraint})", 1)

    lines = bddl_content.split("\n")
    new_bddl = "\n".join(lines[:start_idx]) + "\n" + new_goal + "\n" + "\n".join(lines[end_idx + 1 :])

    extra_match = re.match(r"\(On\s+(\S+)\s+([^\s)]+)\)", extra_constraint)
    if extra_match:
        extra_obj, extra_support = extra_match.groups()
        new_bddl = _append_language_constraint(new_bddl, extra_obj, extra_support)
        new_bddl = _update_obj_of_interest_for_constraint(new_bddl, extra_obj, extra_support)

    return new_bddl


def _change_target_surface(bddl_content: str) -> str:
    """Change goal targets with predicate-aware strategies.

    This method handles multiple goal forms instead of only "(On main_obj target)":
    - Binary predicates: On/In  -> change target, or change object when needed.
    - Unary predicates: Open/Close/Turnon/Turnoff -> change target object/region.
    """
    goal_parse = _parse_goal_section(bddl_content)
    if goal_parse is None:
        return bddl_content

    start_idx, end_idx, goal_text = goal_parse
    constraints = _extract_goal_constraints(goal_text)
    if not constraints:
        return bddl_content

    obj_of_interest = _extract_obj_of_interest_list(bddl_content)
    main_obj = obj_of_interest[0] if obj_of_interest else None

    declared = _extract_declared_entities(bddl_content)
    declared_objects = {name for name in declared if re.search(r"_\d+$", name)}
    fixtures_match = re.search(r"\(:fixtures\s+(.+?)\)\s*\(:objects", bddl_content, re.DOTALL)
    fixture_names = set()
    if fixtures_match:
        fixture_names.update(
            re.findall(r"\b([a-z_][a-z0-9_]*(?:_\d+)?)\s*-\s*[a-z_][a-z0-9_]*\b", fixtures_match.group(1).lower())
        )

    all_regions = sorted(set(re.findall(r"\b([a-z_][a-z0-9_]*_region)\b", bddl_content.lower())))
    all_on_targets = sorted({t for _, _, t in re.findall(r"\((On)\s+([^\s\)]+)\s+([^\s\)]+)\)", bddl_content)})
    all_in_targets = sorted({t for _, _, t in re.findall(r"\((In)\s+([^\s\)]+)\s+([^\s\)]+)\)", bddl_content)})

    def _parse_simple_constraint(constraint: str):
        m = re.match(r"^\((On|In|Open|Close|Turnon|Turnoff)\s+([^\s\)]+)(?:\s+([^\s\)]+))?\)$", constraint.strip())
        return m.groups() if m else None

    def _choose_alt_for_on(subject: str, current: str):
        candidates = sorted(
            {t for t in (all_on_targets + sorted(declared)) if t not in {current, subject, "main_table"}}
        )
        if not candidates:
            return None
        return _det_choice(bddl_content, f"change_target_on_{subject}_{current}", candidates)

    def _choose_alt_for_in(subject: str, current: str):
        def _container_root(token: str):
            m = re.match(r"^([a-z_][a-z0-9_]*_\d+)_.+_region$", token)
            return m.group(1) if m else None

        family_keywords = []
        if "contain_region" in current:
            family_keywords = ["contain_region"]
        elif "heating_region" in current:
            family_keywords = ["heating_region", "cook_region"]
        elif current.endswith(("_top_region", "_middle_region", "_bottom_region")):
            family_keywords = ["_top_region", "_middle_region", "_bottom_region"]
        elif current.endswith("_back_contain_region"):
            family_keywords = ["contain_region", "_back_region", "_front_region"]
        elif current.endswith("_back_region"):
            family_keywords = ["_back_region", "_front_region"]

        base_targets = list(all_in_targets)
        if not base_targets:
            # Conservative fallback: only container-like regions, never init/spawn regions.
            base_targets = [
                r
                for r in all_regions
                if (
                    "contain_region" in r
                    or "heating_region" in r
                    or "cook_region" in r
                    or r.endswith(("_top_region", "_middle_region", "_bottom_region", "_back_region"))
                )
                and ("init_region" not in r)
            ]

        if family_keywords:
            family_filtered = [t for t in base_targets if any(k in t for k in family_keywords)]
            if family_filtered:
                base_targets = family_filtered

        target_candidates = [t for t in sorted(set(base_targets)) if t not in {current, subject, "main_table"}]
        if target_candidates:
            return (
                "target",
                _det_choice(bddl_content, f"change_target_in_target_{subject}_{current}", target_candidates),
            )

        # If there is no alternative container target, change the object target instead.
        # This is useful for single-container tasks (e.g., all "put X in basket" tasks).
        in_triples = re.findall(r"\((In)\s+([^\s\)]+)\s+([^\s\)]+)\)", goal_text)
        in_subjects = {subj for _, subj, _ in in_triples}
        in_container_roots = {_container_root(tgt) for _, _, tgt in in_triples}
        in_container_roots.discard(None)

        object_candidates = [
            o
            for o in sorted(declared_objects)
            if o not in {subject} and o not in in_subjects and o not in in_container_roots
        ]
        if object_candidates:
            return (
                "object",
                _det_choice(bddl_content, f"change_target_in_object_{subject}_{current}", object_candidates),
            )

        # Final safe fallback: use the container object itself as the target if available.
        # Example: (In milk_1 basket_1_contain_region) -> (In milk_1 basket_1)
        # This preserves the task intent better than switching to unrelated init regions.
        current_root = _container_root(current)
        if current_root is not None and current_root != subject:
            return ("target", current_root)
        return None

    def _choose_alt_for_unary(pred: str, operand: str):
        if operand.endswith("_region"):
            # Prefer sibling regions with same owner prefix.
            owner_prefix = operand.rsplit("_", 2)[0] if operand.count("_") >= 2 else None
            sibling_regions = [
                r
                for r in all_regions
                if r != operand
                and (owner_prefix is None or r.startswith(owner_prefix + "_"))
                and ("init_region" not in r)
            ]
            if sibling_regions:
                return _det_choice(
                    bddl_content, f"change_target_{pred.lower()}_region_{operand}", sorted(sibling_regions)
                )

        appliance_like = {"stove", "microwave", "oven", "burner", "light", "lamp"}
        storage_like = {"cabinet", "drawer", "fridge"}
        operand.lower()

        if pred in {"Turnon", "Turnoff"}:
            candidates = [
                f
                for f in sorted(fixture_names)
                if f not in {operand, "main_table"} and any(k in f.lower() for k in appliance_like)
            ]
            if candidates:
                return _det_choice(bddl_content, f"change_target_{pred.lower()}_{operand}", candidates)
            # No alternative appliance in-scene: switch target action polarity on same object.
            return operand

        if pred in {"Open", "Close"}:
            candidates = [
                f
                for f in sorted(fixture_names)
                if f not in {operand, "main_table"} and any(k in f.lower() for k in storage_like)
            ]
            if candidates:
                return _det_choice(bddl_content, f"change_target_{pred.lower()}_{operand}", candidates)
        return None

    def _replace_goal_constraint(old_constraint: str, new_constraint: str):
        if old_constraint == new_constraint:
            return bddl_content
        if new_constraint in constraints:
            return bddl_content

        new_goal = goal_text.replace(old_constraint, new_constraint, 1)
        lines = bddl_content.split("\n")
        return "\n".join(lines[:start_idx]) + "\n" + new_goal + "\n" + "\n".join(lines[end_idx + 1 :])

    # Priority: constraint that directly involves main_obj -> any supported constraint.
    ordered_constraints = []
    for c in constraints:
        parsed = _parse_simple_constraint(c)
        if parsed is None:
            continue
        pred, arg1, _ = parsed
        priority = 0 if (main_obj is not None and arg1 == main_obj) else 1
        ordered_constraints.append((priority, c, parsed))
    ordered_constraints.sort(key=lambda x: x[0])

    for _, old_constraint, parsed in ordered_constraints:
        pred, arg1, arg2 = parsed

        if pred == "On" and arg2 is not None:
            alt = _choose_alt_for_on(arg1, arg2)
            if alt is None:
                continue
            new_constraint = f"(On {arg1} {alt})"
            updated = _replace_goal_constraint(old_constraint, new_constraint)
            if updated != bddl_content:
                updated = _update_language_target(updated, alt)
                updated = _set_language_change_target(updated, new_constraint)
                return _sync_obj_of_interest_with_goal(updated)

        if pred == "In" and arg2 is not None:
            in_choice = _choose_alt_for_in(arg1, arg2)
            if in_choice is None:
                continue
            mode, alt = in_choice
            if mode == "target":
                new_constraint = f"(In {arg1} {alt})"
                updated = _replace_goal_constraint(old_constraint, new_constraint)
                if updated != bddl_content:
                    updated = _set_language_change_target(updated, new_constraint)
                    return _sync_obj_of_interest_with_goal(updated)
            else:
                new_constraint = f"(In {alt} {arg2})"
                updated = _replace_goal_constraint(old_constraint, new_constraint)
                if updated != bddl_content:
                    updated = _set_language_change_target(updated, new_constraint)
                    return _sync_obj_of_interest_with_goal(updated)

        if pred in {"Open", "Close", "Turnon", "Turnoff"}:
            alt_operand = _choose_alt_for_unary(pred, arg1)
            if alt_operand is not None:
                new_pred = pred
                if alt_operand == arg1 and pred == "Turnon":
                    new_pred = "Turnoff"
                elif alt_operand == arg1 and pred == "Turnoff":
                    new_pred = "Turnon"
                new_constraint = f"({new_pred} {alt_operand})"
                updated = _replace_goal_constraint(old_constraint, new_constraint)
                if updated != bddl_content:
                    updated = _set_language_change_target(updated, new_constraint)
                    return _sync_obj_of_interest_with_goal(updated)

    return bddl_content


def _add_negative_constraint(bddl_content: str) -> str:
    """Add negative constraints (what NOT to do).

    E.g., add "(Not (On non_goal_obj goal_location))" to discourage wrong placement.
    """
    goal_parse = _parse_goal_section(bddl_content)
    if goal_parse is None:
        return bddl_content

    start_idx, end_idx, goal_text = goal_parse

    # Extract objects from goal constraints
    constraints = _extract_goal_constraints(goal_text)
    goal_objects = set()
    for constraint in constraints:
        goal_objects.update(_extract_objects_from_constraint(constraint))

    # Try to find non-goal objects from the objects section
    obj_match = re.search(r"(:objects\s+(.+?)(?=:|\)))", bddl_content, re.DOTALL)
    if not obj_match:
        return bddl_content

    # Extract all objects
    all_objects_text = obj_match.group(2)
    all_objects = re.findall(r"\b([a-z_][a-z0-9_]*_\d+)\b", all_objects_text.lower())

    non_goal_objects = [obj for obj in all_objects if obj not in goal_objects]

    if not non_goal_objects:
        return bddl_content

    # Pick one non-goal object and a goal location
    if len(constraints) > 0 and non_goal_objects:
        # For now, just return original
        # Adding negative constraints requires domain support that may not exist
        return bddl_content

    return bddl_content


def rewrite_goal_instruction(
    bddl_content: str,
    perturbation_type: str = "change_target",
    llm_refiner=None,
    *,
    llm_enabled: bool = False,
) -> str:
    """Rewrite BDDL task with goal perturbation.

    Args:
        bddl_content: Full BDDL task definition
        perturbation_type: 'change_target' | 'add_constraint' | 'alternate_location'
        llm_refiner: Optional GeminiInstructionRefiner for semantic validation
        llm_enabled: Whether to use LLM for refinement

    Returns:
        Modified BDDL content
    """

    if perturbation_type == "change_target":
        result = _change_target_surface(bddl_content)
    elif perturbation_type == "add_constraint":
        result = _add_ordering_constraint(bddl_content)
    elif perturbation_type == "alternate_location":
        result = _change_target_surface(bddl_content)
    else:
        result = bddl_content

    return result


def rewrite_goal_variants(
    bddl_content: str,
    num_variants: int = 2,
    llm_refiner=None,
    *,
    llm_enabled: bool = False,
) -> dict[str, str]:
    """Generate multiple goal variants.

    Args:
        bddl_content: Full BDDL task definition
        num_variants: Number of variants to generate (max 2)
        llm_refiner: Optional LLM refiner for validation
        llm_enabled: Whether to enable LLM validation

    Returns:
        Dict mapping variant name to modified BDDL content
        (Keys are just 'change_target', 'add_constraint' without 'goal_' prefix)
    """
    perturbation_types = ["change_target", "add_constraint"][:num_variants]

    return {
        ptype: rewrite_goal_instruction(
            bddl_content,
            ptype,
            llm_refiner,
            llm_enabled=llm_enabled,
        )
        for ptype in perturbation_types
    }
