"""
Object Composition Perturbation Rewriter for LIBERO BDDL tasks.

Modifies the `:objects` field by:
- Adding distractor objects (same type from library)

Reference: LIBERO-Pro object perturbation design.
"""

import hashlib
import re
from typing import List, Dict, Set, Tuple, Optional


# Object type to available instances mapping
# (In real usage, would load from LIBERO asset library)
OBJECT_VARIANTS = {
    'akita_black_bowl': ['akita_black_bowl_1', 'akita_black_bowl_2', 'akita_black_bowl_3'],
    'cookies': ['cookies_1', 'cookies_2'],
    'glazed_rim_porcelain_ramekin': ['glazed_rim_porcelain_ramekin_1', 'glazed_rim_porcelain_ramekin_2'],
    'plate': ['plate_1', 'plate_2', 'plate_3'],
    'wooden_block_flat': ['wooden_block_flat_1', 'wooden_block_flat_2'],
    'wooden_block_triangle': ['wooden_block_triangle_1', 'wooden_block_triangle_2'],
    'stacked_block_pyramid': ['stacked_block_pyramid_1', 'stacked_block_pyramid_2'],
    'glass': ['glass_1', 'glass_2'],
    'utensil_spatula': ['utensil_spatula_1', 'utensil_spatula_2'],
    'utensil_spoon': ['utensil_spoon_1', 'utensil_spoon_2'],
}


def _det_choice(seed_text: str, key: str, options: List) -> any:
    """Deterministically pick one option using SHA256 hash."""
    hash_str = hashlib.sha256(f"{seed_text}_{key}".encode()).hexdigest()
    choice_idx = int(hash_str, 16) % len(options)
    return options[choice_idx]


def _get_object_type(obj_name: str) -> str:
    """Extract object type from instance name.
    
    E.g., 'akita_black_bowl_1' -> 'akita_black_bowl'
    """
    parts = obj_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return obj_name


def _parse_objects_section(bddl_content: str) -> Optional[Tuple[int, int, str]]:
    """Extract objects section from BDDL.
    
    Returns:
        (start_line_idx, end_line_idx, objects_text) or None
    """
    lines = bddl_content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if '(:objects' in line:
            start_idx = i
        elif start_idx is not None and line.strip().startswith(')'):
            end_idx = i
            break
    
    if start_idx is not None and end_idx is not None:
        objects_text = '\n'.join(lines[start_idx:end_idx+1])
        return (start_idx, end_idx, objects_text)
    
    return None


def _parse_goal_section(bddl_content: str) -> Optional[Tuple[int, int, str]]:
    """Extract goal section from BDDL."""
    lines = bddl_content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if '(:goal' in line:
            start_idx = i
        elif start_idx is not None and line.strip() == ')':
            # Check if this closes the goal block
            goal_content = '\n'.join(lines[start_idx:i+1])
            if goal_content.count('(') == goal_content.count(')'):
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        goal_text = '\n'.join(lines[start_idx:end_idx+1])
        return (start_idx, end_idx, goal_text)
    
    return None


def _parse_init_section(bddl_content: str) -> Optional[Tuple[int, int, str]]:
    """Extract init section from BDDL."""
    lines = bddl_content.split('\n')
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '(:init' in line:
            start_idx = i
        elif start_idx is not None and line.strip() == ')':
            init_content = '\n'.join(lines[start_idx:i+1])
            if init_content.count('(') == init_content.count(')'):
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        init_text = '\n'.join(lines[start_idx:end_idx+1])
        return (start_idx, end_idx, init_text)

    return None


def _replace_exact_symbol(text: str, old: str, new: str) -> str:
    return re.sub(rf'(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])', new, text)


def _extract_existing_indices(current_objects: List[str]) -> Dict[str, Set[int]]:
    indices: Dict[str, Set[int]] = {}
    for obj in current_objects:
        parts = obj.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            obj_type = parts[0]
            indices.setdefault(obj_type, set()).add(int(parts[1]))
    return indices


def _choose_distractor_support(bddl_content: str, init_text: str, goal_objects: Set[str], distractor_obj: str) -> Optional[str]:
    regions = []
    for match in re.finditer(r'\(\s*([a-zA-Z0-9_]+)\s*\n\s*\(:target\s+([a-zA-Z0-9_]+)\)', bddl_content):
        regions.append(f'{match.group(2)}_{match.group(1)}')

    used_regions = set()
    for line in init_text.splitlines():
        match = re.search(r'\(On\s+(\S+)\s+(\S+)\)', line.strip())
        if match:
            used_regions.add(match.group(2))
            
    support_candidates = list(set(regions) - used_regions)

    if not support_candidates:
        return None
    support_candidates.sort()
    return _det_choice(init_text, 'distractor_support', support_candidates)


def _extract_goal_objects(goal_text: str) -> Set[str]:
    """Extract all referenced object names from goal section."""
    # Find all capitalized words that look like object references
    matches = re.findall(r'\b([a-z_0-9]+)\b', goal_text.lower())
    return set(matches)


def _add_distractor_objects(
    bddl_content: str,
    num_distractors: int = 1
) -> str:
    """Add distractor objects to the objects section while preserving goal-critical objects."""
    
    parse_result = _parse_objects_section(bddl_content)
    init_parse = _parse_init_section(bddl_content)
    if parse_result is None:
        return bddl_content
    
    start_idx, end_idx, objects_text = parse_result
    
    # Also get goal objects to know which we can't remove
    goal_parse = _parse_goal_section(bddl_content)
    goal_objects = set()
    if goal_parse:
        _, _, goal_text = goal_parse
        goal_objects = _extract_goal_objects(goal_text)
    
    # Extract currently listed objects
    lines = bddl_content.split('\n')
    objects_lines = lines[start_idx+1:end_idx]
    current_objects = []
    
    for line in objects_lines:
        # Parse "obj1 obj2 - type" format
        if '-' in line:
            obj_part = line.split('-')[0].strip()
            current_objects.extend(obj_part.split())
    
    # Collect types from existing objects for candidate distractors
    available_types = {}
    for obj in current_objects:
        obj_type = _get_object_type(obj)
        if obj_type not in available_types:
            available_types[obj_type] = []
        if obj not in available_types[obj_type]:
            available_types[obj_type].append(obj)
    
    # Select distractor type and instance
    if not available_types:
        return bddl_content
    
    selected_type = _det_choice(bddl_content, 'distractor_type', list(available_types.keys()))
    candidates = available_types[selected_type]
    existing_indices = _extract_existing_indices(candidates).get(selected_type, set())
    
    # Find next available index
    next_idx = max(existing_indices) + 1 if existing_indices else 3
    distractor_obj = f"{selected_type}_{next_idx}"
    
    # Build new objects section with distractor
    new_objects_lines = []
    type_found = False
    for line in objects_text.splitlines():
        if '-' in line and not type_found:
            parts = line.split('-')
            if parts[1].strip() == selected_type:
                # Insert the new object right before the hyphen
                line = f"{parts[0].rstrip()} {distractor_obj} - {parts[1].strip()}"
                type_found = True
        new_objects_lines.append(line)
    
    if type_found:
        new_objects_text = '\n'.join(new_objects_lines)
    else:
        new_objects_text = objects_text.rstrip(')').rstrip() + f"\n    {distractor_obj} - {selected_type}\n  )"
    
    new_bddl = bddl_content[:bddl_content.find(objects_text)]
    new_bddl += new_objects_text
    new_bddl += bddl_content[bddl_content.find(objects_text) + len(objects_text):]

    if init_parse is None:
        return new_bddl

    _, _, init_text = _parse_init_section(new_bddl) or (None, None, None)
    if not init_text:
        return new_bddl

    support = _choose_distractor_support(new_bddl, init_text, goal_objects, distractor_obj)
    if support is None:
        return new_bddl

    new_init_text = init_text.rstrip(')').rstrip() + f"\n    (On {distractor_obj} {support})\n  )"
    new_bddl = new_bddl.replace(init_text, new_init_text, 1)
    
    return new_bddl




def rewrite_object_instruction(
    bddl_content: str,
    perturbation_type: str = 'add_distractor'
) -> str:
    """Rewrite BDDL task with object perturbation.
    
    Args:
        bddl_content: Full BDDL task definition
        perturbation_type: 'add_distractor'
    
    Returns:
        Modified BDDL content
    """
    
    if perturbation_type == 'add_distractor':
        return _add_distractor_objects(bddl_content, num_distractors=1)
    else:
        return bddl_content


def rewrite_object_variants(bddl_content: str, num_variants: int = 1) -> Dict[str, str]:
    """Generate multiple object variants.
    
    Args:
        bddl_content: Full BDDL task definition
        num_variants: Number of variants to generate (max 1)
    
    Returns:
        Dict mapping variant name to modified BDDL content
        (Keys are just 'add_distractor' without 'object_' prefix)
    """
    perturbation_types = ['add_distractor'][:num_variants]
    return {
        ptype: rewrite_object_instruction(bddl_content, ptype)
        for ptype in perturbation_types
    }
