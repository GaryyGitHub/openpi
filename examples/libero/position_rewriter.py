"""Position/Layout Perturbation Rewriter for LIBERO BDDL tasks.

Modifies the `:regions` field by applying a deterministic spatial shift.
"""

import hashlib
import re
from typing import Tuple, List, Dict, Optional


def _fmt_coord(value: float) -> str:
    """Format coordinates with enough precision to preserve small perturbations."""
    text = f"{value:.6f}".rstrip('0').rstrip('.')
    if text == "-0":
        return "0"
    return text


def _det_choice(seed_text: str, key: str, options: List) -> any:
    """Deterministically pick one option using SHA256 hash."""
    hash_str = hashlib.sha256(f"{seed_text}_{key}".encode()).hexdigest()
    choice_idx = int(hash_str, 16) % len(options)
    return options[choice_idx]


def _parse_regions(bddl_content: str) -> Optional[Tuple[int, int, str]]:
    """Extract regions block from BDDL content.
    
    Returns:
        (start_line, end_line, regions_text) or None if not found
    """
    lines = bddl_content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if '(:regions' in line:
            start_idx = i
        if start_idx is not None and line.strip() == ')' and i > start_idx:
            # Check if this closing paren closes the regions block
            # by counting nested parens
            region_content = '\n'.join(lines[start_idx:i+1])
            if region_content.count('(') == region_content.count(')'):
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        regions_text = '\n'.join(lines[start_idx:end_idx+1])
        return (start_idx, end_idx, regions_text)
    
    return None


def _extract_coordinate_ranges(region_text: str) -> List[Tuple[float, float, float, float]]:
    """Extract (x1, y1, x2, y2) tuples from region's `:ranges` line.

    LIBERO expects x2 >= x1 and y2 >= y1.
    """
    # Match (:ranges ((...)))
    match = re.search(r'\(:ranges\s*\(([\d\s\.\-]+)\)', region_text)
    if not match:
        return []
    
    coords_str = match.group(1).strip()
    # Split by closing/opening parens and whitespace
    parts = re.findall(r'[\d\.\-]+', coords_str)
    
    ranges = []
    for i in range(0, len(parts), 4):
        if i + 3 < len(parts):
            ranges.append((
                float(parts[i]),
                float(parts[i+1]),
                float(parts[i+2]),
                float(parts[i+3])
            ))
    
    return ranges


def _apply_position_perturbation(
    region_text: str,
    perturbation_type: str,
    seed: str
) -> str:
    """Apply position perturbation to a single region definition.
    
    Args:
        region_text: Single region block text (multiline)
        perturbation_type: 'shift'
        seed: Deterministic seed
    
    Returns:
        Modified region text
    """
    ranges_match = re.search(r'\(:ranges\s*\(([^)]+)\)', region_text)
    if not ranges_match:
        return region_text
    
    ranges_content = ranges_match.group(1).strip()
    parts = re.findall(r'[\d\.\-]+', ranges_content)
    
    if len(parts) < 4:
        return region_text
    
    # Parse all coordinate ranges in this region as (x1, y1, x2, y2)
    ranges = []
    for i in range(0, len(parts), 4):
        if i + 3 < len(parts):
            ranges.append((
                float(parts[i]),
                float(parts[i+1]),
                float(parts[i+2]),
                float(parts[i+3])
            ))
    
    # Apply perturbation
    perturbed_ranges = []
    for x1, y1, x2, y2 in ranges:
        width = x2 - x1
        height = y2 - y1

        if width < 0 or height < 0:
            # Defensive fallback: normalize malformed inputs before perturbation.
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            width = x2 - x1
            height = y2 - y1
        
        if perturbation_type == 'shift':
            # ±10% workspace shift
            shift_mag = _det_choice(f"{seed}_{x1}_{y1}_shift", "shift_mag", [0.04, 0.03])
            x_shift = _det_choice(f"{seed}_{x1}_shift_x", "shift_x", [-shift_mag, shift_mag])
            y_shift = _det_choice(f"{seed}_{y1}_shift_y", "shift_y", [-shift_mag, shift_mag])
            perturbed = (x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift)
        
        else:
            perturbed = (x1, y1, x2, y2)

        # Guarantee valid rectangle ordering required by LIBERO assertion.
        px1, py1, px2, py2 = perturbed
        perturbed_ranges.append((min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)))
    
    # Reconstruct ranges line
    ranges_str = ' '.join(
        f"({_fmt_coord(r[0])} {_fmt_coord(r[1])} {_fmt_coord(r[2])} {_fmt_coord(r[3])})"
        for r in perturbed_ranges
    )
    
    # Use ranges_match.end() (not end(1)) to skip the original coord ")" consumed by the regex,
    # preventing a duplicate closing paren in the output.
    perturbed_text = region_text[:ranges_match.start(1)] + ranges_str + region_text[ranges_match.end():]
    
    return perturbed_text


def rewrite_position_instruction(
    bddl_content: str,
    perturbation_type: str = 'shift'
) -> str:
    """Rewrite BDDL task with position perturbation.
    
    Args:
        bddl_content: Full BDDL task definition
        perturbation_type: only 'shift' is supported
    
    Returns:
        Modified BDDL content
    """
    if perturbation_type != 'shift':
        raise ValueError(f"Unsupported position perturbation: {perturbation_type}. Only 'shift' is supported.")

    region_parse = _parse_regions(bddl_content)
    if region_parse is None:
        return bddl_content
    
    start_idx, end_idx, regions_text = region_parse
    lines = bddl_content.split('\n')
    
    # Split regions into individual region definitions
    # Find all "(" and matching ")" to extract each region
    region_lines = lines[start_idx:end_idx+1]
    
    perturbed_lines = []
    i = 0
    while i < len(region_lines):
        line = region_lines[i]
        
        if '(:regions' in line:
            perturbed_lines.append(line)
            i += 1
        elif '(' in line and 'regions' not in line:
            # Start of a single region definition
            region_block = []
            paren_count = 0
            j = i
            
            while j < len(region_lines):
                region_block.append(region_lines[j])
                paren_count += region_lines[j].count('(') - region_lines[j].count(')')
                
                if paren_count == 0 and len(region_block) > 1:
                    break
                j += 1
            
            region_text = '\n'.join(region_block)
            perturbed_region = _apply_position_perturbation(region_text, perturbation_type, bddl_content)
            
            perturbed_lines.extend(perturbed_region.split('\n'))
            i = j + 1
        else:
            perturbed_lines.append(line)
            i += 1
    
    # Reconstruct full BDDL
    perturbed_regions = '\n'.join(perturbed_lines)
    result = '\n'.join(lines[:start_idx]) + '\n' + perturbed_regions + '\n' + '\n'.join(lines[end_idx+1:])
    
    return result


def rewrite_position_variants(bddl_content: str) -> Dict[str, str]:
    """Generate supported position variants.

    Returns a single `shift` variant that translates all task regions together.
    """
    return {
        'shift': rewrite_position_instruction(bddl_content, 'shift')
    }
