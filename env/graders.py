"""
Graders for each task.
All graders return a float in [0.0, 1.0].
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.caravan_env import CaravanEnv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _feasibility_score(env: "CaravanEnv") -> float:
    total = env.total_items
    placed = len(env.placed_items)
    if total == 0:
        return 0.0

    all_cells = {}
    overlap_count = 0

    for pi in env.placed_items:
        for cell in pi.cells():
            if cell in all_cells:
                overlap_count += 1
            all_cells[cell] = pi.item.id

    placement_ratio = placed / total
    overlap_penalty = min(1.0, overlap_count / max(1, placed * 5))

    return max(0.0, placement_ratio - overlap_penalty)


def _weight_balance_score(env: "CaravanEnv") -> float:
    if not env.placed_items:
        return 0.0

    cx = env.grid_width / 2
    cy = env.grid_height / 2

    left_w = right_w = front_w = rear_w = 0.0

    for pi in env.placed_items:
        item_cx = pi.x + pi.effective_width / 2
        item_cy = pi.y + pi.effective_height / 2
        w = pi.item.weight_kg

        if item_cx < cx:
            left_w += w
        else:
            right_w += w

        if item_cy < cy:
            front_w += w
        else:
            rear_w += w

    total_w = left_w + right_w
    if total_w == 0:
        return 0.0

    lr_imbalance = abs(left_w - right_w) / total_w
    fb_imbalance = abs(front_w - rear_w) / total_w

    lr_score = max(0.0, 1.0 - lr_imbalance * 2)
    fb_score = max(0.0, 1.0 - fb_imbalance * 1.5)

    return (lr_score + fb_score) / 2


def _space_utilisation_score(env: "CaravanEnv") -> float:
    total_cells = env.grid_width * env.grid_height
    used_cells = sum(len(pi.cells()) for pi in env.placed_items)

    ratio = used_cells / total_cells

    if ratio <= 0.70:
        return ratio / 0.70
    else:
        return max(0.0, 1.0 - (ratio - 0.70) * 5)


def _zone_coherence_score(env: "CaravanEnv") -> float:
    if not env.placed_items:
        return 0.0

    front_zone_items = {"kitchen", "dining_table", "sofa", "fridge"}
    rear_zone_items = {"bed", "bathroom", "wardrobe", "storage"}

    correct = 0
    total = 0
    mid_y = env.grid_height / 2

    for pi in env.placed_items:
        item_cy = pi.y + pi.effective_height / 2
        itype = pi.item.item_type.value

        if itype in front_zone_items:
            total += 1
            if item_cy < mid_y:
                correct += 1

        elif itype in rear_zone_items:
            total += 1
            if item_cy >= mid_y:
                correct += 1

    return correct / total if total > 0 else 0.5


def _aisle_score(env: "CaravanEnv") -> float:
    aisle_cols = set(range(13, 17))
    aisle_cells = len(aisle_cols) * env.grid_height

    occupied = set()
    for pi in env.placed_items:
        occupied.update(pi.cells())

    blocked = sum(1 for (x, y) in occupied if x in aisle_cols)

    return max(0.0, 1.0 - blocked / aisle_cells)


# ---------------------------------------------------------------------------
# NEW: Priority-based Accessibility (WINNING FEATURE)
# ---------------------------------------------------------------------------

def _priority_accessibility_score(env: "CaravanEnv") -> float:
    if not env.placed_items:
        return 0.0

    occupied = set()
    for pi in env.placed_items:
        occupied.update(pi.cells())

    total_weight = 0
    score = 0

    for pi in env.placed_items:
        priority = pi.item.accessibility_priority
        weight = 4 - priority  # 1→3, 2→2, 3→1

        total_weight += weight

        is_accessible = False

        for (x, y) in pi.cells():
            neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

            for nx, ny in neighbors:
                if (
                    0 <= nx < env.grid_width and
                    0 <= ny < env.grid_height and
                    (nx, ny) not in occupied
                ):
                    is_accessible = True
                    break

            if is_accessible:
                break

        if is_accessible:
            score += weight

    return score / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

def grade_easy(env: "CaravanEnv") -> float:
    feas = _feasibility_score(env)
    all_placed = len(env.placed_items) == env.total_items
    bonus = 0.1 if all_placed else 0.0
    return min(1.0, feas + bonus)


def grade_medium(env: "CaravanEnv") -> float:
    feas = _feasibility_score(env)
    if feas < 0.5:
        return feas * 0.4

    wb = _weight_balance_score(env)
    su = _space_utilisation_score(env)

    return 0.40 * feas + 0.30 * wb + 0.30 * su


def grade_hard(env: "CaravanEnv") -> float:
    feas = _feasibility_score(env)
    if feas < 0.3:
        return feas * 0.25

    wb = _weight_balance_score(env)
    su = _space_utilisation_score(env)
    zs = _zone_coherence_score(env)
    ai = _aisle_score(env)
    ac = _priority_accessibility_score(env)

    score = (
        0.22 * feas
        + 0.18 * wb
        + 0.18 * su
        + 0.16 * zs
        + 0.12 * ai
        + 0.14 * ac
    )

    return max(0.0, min(1.0, score))


GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}