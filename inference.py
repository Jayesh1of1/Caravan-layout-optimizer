"""
inference.py — Iterative Step-by-Step Inference for Caravan Layout Optimizer.

Architecture:
  For each item remaining:
    1. Ask LLM: "given current grid state, where should I place THIS item?"
    2. Execute placement
    3. If placement FAILS (overlap/out-of-bounds) → show LLM the error, retry up to MAX_RETRIES
    4. If still fails → heuristic fallback for that item only
    5. Move to next item with fully updated state (real feedback loop)

Required env vars:
    API_BASE_URL   LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
"""

import os
import json
import time
import requests
import sys
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_RETRIES = 3   # retries per item if placement rejected

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def safe_request(method: str, url: str, **kwargs):
    try:
        r = requests.request(method, url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [API ERROR] {e}")
        return None


def env_reset(task_id: str) -> dict:
    return safe_request("POST", f"{ENV_BASE_URL}/reset", json={"task_id": task_id}) or {}


def env_step(item_id: str, x: int, y: int, rotation: int = 0) -> dict:
    return safe_request(
        "POST", f"{ENV_BASE_URL}/step",
        json={"item_id": item_id, "x": x, "y": y, "rotation": rotation}
    ) or {}


def env_state() -> dict:
    return safe_request("GET", f"{ENV_BASE_URL}/state") or {}

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def occupied_cells(state: dict) -> set:
    """Return set of (x, y) cells currently occupied by placed items."""
    cells = set()
    for p in state.get("placed_items", []):
        w = p.get("effective_width",  p["item"]["width"])
        h = p.get("effective_height", p["item"]["height"])
        for dx in range(w):
            for dy in range(h):
                cells.add((p["x"] + dx, p["y"] + dy))
    return cells


def render_placed_summary(state: dict) -> str:
    """Human-readable list of placed items with exact occupied ranges."""
    lines = []
    for p in state.get("placed_items", []):
        w = p.get("effective_width",  p["item"]["width"])
        h = p.get("effective_height", p["item"]["height"])
        x2 = p["x"] + w - 1
        y2 = p["y"] + h - 1
        lines.append(
            f"  {p['item']['id']:15s}  x={p['x']:2d}-{x2:2d}  y={p['y']:2d}-{y2:2d}"
            f"  rot={p['rotation']}deg  ({p['item']['weight_kg']}kg)"
        )
    return "\n".join(lines) if lines else "  (none yet)"


def render_free_summary(state: dict) -> str:
    """Count free cells in each quadrant of the grid."""
    occ = occupied_cells(state)
    sections = {
        "LEFT-FRONT  (x 0-12,  y 0-7) ": [(x, y) for x in range(0,  13) for y in range(0,  8)],
        "RIGHT-FRONT (x 17-29, y 0-7) ": [(x, y) for x in range(17, 30) for y in range(0,  8)],
        "LEFT-REAR   (x 0-12,  y 8-14)": [(x, y) for x in range(0,  13) for y in range(8,  15)],
        "RIGHT-REAR  (x 17-29, y 8-14)": [(x, y) for x in range(17, 30) for y in range(8,  15)],
    }
    lines = []
    for label, region in sections.items():
        free  = sum(1 for c in region if c not in occ)
        total = len(region)
        lines.append(f"  {label}: {free}/{total} cells free")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert caravan interior layout designer with deep spatial reasoning skills.

CARAVAN GRID: 30 columns (x: 0-29) x 15 rows (y: 0-14).  1 cell = 20 cm.
  - Front of caravan (hitching end) = y = 0
  - Rear  of caravan (sleeping end) = y = 14
  - Central aisle = columns 13-16 (keep mostly clear for walking)

HARD CONSTRAINTS (never violate):
  1. x + effective_width  <= 30
  2. y + effective_height <= 15
  3. Item must NOT overlap any placed item (check occupied ranges carefully)
  4. rotation 90 swaps width and height

SOFT GOALS (optimise for):
  - Kitchen, dining table, sofa, fridge  -> FRONT zone (item centre y < 7.5)
  - Bed, bathroom, wardrobe, storage     -> REAR  zone (item centre y >= 7.5)
  - Place items LEFT (x<=12) or RIGHT (x>=17) to preserve aisle cols 13-16
  - Balance heavy items left-right (bed 40kg vs bathroom 50kg)
  - Priority-1 items need at least one free adjacent cell

YOUR TASK: Place exactly ONE item per response.
Think carefully:
  1. Read occupied ranges to find conflict-free space.
  2. Choose x, y within the preferred zone.
  3. Double-check: x + w <= 30, y + h <= 15, no overlap.

Return ONLY this JSON (no explanation, no markdown):
{"item_id": "...", "x": <int>, "y": <int>, "rotation": <0 or 90>}
"""


def build_step_prompt(
    item: dict,
    state: dict,
    last_error: str = None,
    retry_num: int = 0,
) -> str:
    """Build the per-item placement prompt with full state context."""
    w, h     = item["width"], item["height"]
    metrics  = state.get("metrics", {})
    score    = state.get("score", 0.0)
    unplaced = state.get("unplaced_items", [])
    remaining_after = [u["id"] for u in unplaced if u["id"] != item["id"]]

    prompt = f"""
ITEM TO PLACE NOW:
  id           = {item['id']}
  normal size  = {w} x {h} cells  ({w*20} x {h*20} cm)
  rotated size = {h} x {w} cells  ({h*20} x {w*20} cm)
  weight       = {item['weight_kg']} kg
  zone pref    = {item['zone_preference']}
  priority     = {item['accessibility_priority']} (1=high needs access, 3=low)

CURRENT SCORE: {score:.4f}
METRICS: feasibility={metrics.get('feasibility',0):.3f}  weight_balance={metrics.get('weight_balance',0):.3f}  space={metrics.get('space_utilisation',0):.3f}  zone={metrics.get('zone_coherence',0):.3f}  aisle={metrics.get('aisle_score',0):.3f}

PLACED ITEMS (do NOT overlap these):
{render_placed_summary(state)}

FREE SPACE BY QUADRANT:
{render_free_summary(state)}

ITEMS STILL TO PLACE AFTER THIS:
{', '.join(remaining_after) if remaining_after else '(this is the LAST item)'}
"""

    if last_error and retry_num > 0:
        prompt += f"""
*** RETRY {retry_num}/{MAX_RETRIES} — PREVIOUS ATTEMPT FAILED ***
  Reason: {last_error}
  You MUST pick DIFFERENT coordinates that avoid this error.
  Re-read the placed item ranges above VERY carefully.
"""

    prompt += "\nReturn ONLY the JSON object."
    return prompt.strip()

# ---------------------------------------------------------------------------
# LLM call — single item
# ---------------------------------------------------------------------------

def llm_place_item(
    item: dict,
    state: dict,
    conversation_history: list,
    last_error: str = None,
    retry_num: int = 0,
) -> dict:
    """
    Ask LLM where to place `item` given current `state`.
    Uses multi-turn conversation_history so LLM remembers earlier placements.
    Returns validated action dict, or None on any failure.
    """
    user_msg = build_step_prompt(item, state, last_error, retry_num)
    conversation_history.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
            temperature=0.1,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # Extract JSON object
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in response: {raw!r}")

        action = json.loads(raw[start:end])

        for field in ("item_id", "x", "y"):
            if field not in action:
                raise ValueError(f"Missing field '{field}' in response")

        # Enforce correct item_id (LLM must not hallucinate a different one)
        action["item_id"]  = item["id"]
        action["x"]        = int(action["x"])
        action["y"]        = int(action["y"])
        action["rotation"] = int(action.get("rotation", 0))

        # Client-side bounds pre-check
        rot = action["rotation"]
        ew  = item["height"] if rot == 90 else item["width"]
        eh  = item["width"]  if rot == 90 else item["height"]
        gw  = state.get("grid_width",  30)
        gh  = state.get("grid_height", 15)

        if action["x"] < 0 or action["y"] < 0:
            raise ValueError(f"Negative coordinates x={action['x']}, y={action['y']}")
        if action["x"] + ew > gw:
            raise ValueError(f"Exceeds grid width: x={action['x']}+w={ew} > {gw}")
        if action["y"] + eh > gh:
            raise ValueError(f"Exceeds grid height: y={action['y']}+h={eh} > {gh}")

        # Client-side overlap pre-check
        new_cells = {(action["x"]+dx, action["y"]+dy) for dx in range(ew) for dy in range(eh)}
        overlap   = new_cells & occupied_cells(state)
        if overlap:
            raise ValueError(f"Would overlap placed items at {list(overlap)[:3]}")

        # Success — store assistant reply in history
        conversation_history.append({"role": "assistant", "content": raw})
        return action

    except Exception as e:
        # Record the failure in conversation so LLM learns from it
        conversation_history.append({
            "role": "assistant",
            "content": f"[INVALID RESPONSE — error: {e}]"
        })
        return None

# ---------------------------------------------------------------------------
# Heuristic fallback — single item
# ---------------------------------------------------------------------------

def heuristic_place_item(item: dict, state: dict) -> dict:
    """
    Scan-based fallback: find the first non-overlapping valid position
    for a single item. Tries zone-preferred region first, then full grid.
    """
    occ   = occupied_cells(state)
    gw    = state.get("grid_width",  30)
    gh    = state.get("grid_height", 15)
    pref  = item.get("zone_preference", "any")
    itype = item.get("item_type", "")
    front_types = {"kitchen", "dining_table", "sofa", "fridge"}

    for rotation in [0, 90]:
        ew = item["height"] if rotation == 90 else item["width"]
        eh = item["width"]  if rotation == 90 else item["height"]

        if ew > gw or eh > gh:
            continue

        # Zone-aware y-ranges (preferred first, then full grid)
        if itype in front_types or pref == "front":
            y_ranges = [range(0, max(1, gh//2 - eh + 1)), range(gh//2, gh - eh + 1)]
        elif pref == "rear":
            y_ranges = [range(gh//2, gh - eh + 1), range(0, max(1, gh//2 - eh + 1))]
        else:
            y_ranges = [range(0, gh - eh + 1)]

        # x-ranges: left of aisle, right of aisle, then anywhere
        x_ranges = [
            range(0,  min(13, gw - ew + 1)),
            range(17, gw - ew + 1),
            range(0,  gw - ew + 1),
        ]

        for y_range in y_ranges:
            for x_range in x_ranges:
                for y in y_range:
                    for x in x_range:
                        cells = {(x+dx, y+dy) for dx in range(ew) for dy in range(eh)}
                        if not (cells & occ):
                            return {"item_id": item["id"], "x": x, "y": y, "rotation": rotation}

    return None  # grid is genuinely full

# ---------------------------------------------------------------------------
# Core iterative episode loop
# ---------------------------------------------------------------------------

def run_task(task_id: str, global_start: float) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    state = env_reset(task_id)
    if not state:
        print("  [ERROR] Could not reset env")
        return 0.0

    unplaced_start = state.get("unplaced_items", [])
    print(f"  Items: {[u['id'] for u in unplaced_start]}")

    # Shared multi-turn conversation — LLM sees ALL prior placements
    conversation_history: list = []
    step_num   = 0
    prev_score = 0.0

    while True:
        # Re-read state each iteration (real feedback loop)
        state = env_state()
        if not state or state.get("done"):
            break

        unplaced = state.get("unplaced_items", [])
        if not unplaced:
            break

        # Time budget check
        elapsed = time.time() - global_start
        if elapsed > 900:
            print("  [TIMEOUT] Approaching 20-min limit, stopping early")
            break

        # Pick highest-priority unplaced item
        item = min(unplaced, key=lambda u: (u.get("accessibility_priority", 3), u["id"]))
        step_num += 1
        print(f"\n  Step {step_num}: '{item['id']}'  "
              f"({item['width']}x{item['height']}, {item['weight_kg']}kg, "
              f"zone={item['zone_preference']}) ...")

        placed_ok  = False
        last_error = None

        # ── Per-item iterative retry loop ──────────────────────────────────
        for attempt in range(MAX_RETRIES + 1):

            if attempt > 0:
                print(f"    Retry {attempt}/{MAX_RETRIES}: feeding error back to LLM ...")

            action = llm_place_item(
                item, state, conversation_history,
                last_error=last_error,
                retry_num=attempt,
            )

            if action is None:
                last_error = "LLM response failed client-side validation"
                print(f"    Attempt {attempt+1}: client-side check failed")
                continue

            # Execute in environment
            result = env_step(action["item_id"], action["x"], action["y"], action["rotation"])

            if not result:
                last_error = "No response from environment API"
                continue

            info = result.get("info", {})

            if "placed" in info:
                # ✅ Accepted
                new_state = result.get("state", {})
                new_score = new_state.get("score", 0.0)
                reward    = result.get("reward", 0.0)
                delta     = new_score - prev_score

                print(f"    OK  x={action['x']:2d} y={action['y']:2d} "
                      f"rot={action['rotation']:2d}  "
                      f"reward={reward:+.3f}  score={new_score:.4f} "
                      f"({delta:+.4f})")

                # Inject success feedback into conversation history
                conversation_history.append({
                    "role": "user",
                    "content": (
                        f"ACCEPTED: {item['id']} placed at x={action['x']}, "
                        f"y={action['y']}, rotation={action['rotation']}. "
                        f"Reward={reward:+.3f}. Score is now {new_score:.4f}."
                    )
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": "Understood. I will avoid that region for future items."
                })

                prev_score = new_score
                placed_ok  = True
                state      = new_state
                break

            else:
                # ❌ Rejected by environment
                last_error = (
                    info.get("invalid_reason")
                    or info.get("warning")
                    or "Unknown rejection"
                )
                print(f"    FAIL attempt {attempt+1}: {last_error}")

                # Inject rejection into conversation — LLM learns from it
                conversation_history.append({
                    "role": "user",
                    "content": (
                        f"REJECTED by environment: {last_error}. "
                        f"Do NOT use x={action['x']}, y={action['y']} for {item['id']}. "
                        f"Pick a completely different position."
                    )
                })

        # ── Heuristic fallback if all LLM attempts failed ──────────────────
        if not placed_ok:
            print(f"    All LLM attempts failed — using heuristic fallback ...")
            fb = heuristic_place_item(item, state)

            if fb:
                result = env_step(fb["item_id"], fb["x"], fb["y"], fb["rotation"])
                if result and "placed" in result.get("info", {}):
                    new_score = result.get("state", {}).get("score", 0.0)
                    print(f"    FALLBACK OK  x={fb['x']} y={fb['y']} "
                          f"rot={fb['rotation']}  score={new_score:.4f}")
                    state = result.get("state", state)
                else:
                    print(f"    FALLBACK also rejected for '{item['id']}'")
            else:
                print(f"    No valid position exists for '{item['id']}' — grid full?")

        if state.get("done"):
            break

    # Final result
    final   = env_state()
    score   = final.get("score", 0.0)
    metrics = final.get("metrics", {})

    print(f"\n  {'─'*50}")
    print(f"  FINAL SCORE : {score:.4f}")
    for k, v in metrics.items():
        if isinstance(v, float) and v <= 1.0:
            bar = "█" * int(v * 20)
            print(f"    {k:22s} {v:.4f}  {bar}")
        else:
            print(f"    {k:22s} {v}")

    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*62)
    print("   Caravan Layout Optimizer — Iterative LLM Inference")
    print("="*62)
    print(f"  Model : {MODEL_NAME}")
    print(f"  API   : {API_BASE_URL}")
    print(f"  Env   : {ENV_BASE_URL}")
    print()

    try:
        r = requests.get(f"{ENV_BASE_URL}/", timeout=10)
        r.raise_for_status()
        print(f"Environment online: {r.json().get('status', 'ok')} OK\n")
    except Exception as e:
        print(f"[FATAL] Environment unreachable: {e}")
        return

    tasks        = ["task_easy", "task_medium", "task_hard"]
    scores       = {}
    global_start = time.time()

    for task_id in tasks:
        if time.time() - global_start > 1100:
            print("[TIMEOUT] Skipping remaining tasks")
            break
        scores[task_id] = run_task(task_id, global_start)

    print(f"\n{'='*62}")
    print("  FINAL RESULTS")
    print(f"{'='*62}")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 30)
        print(f"  {tid:15s}  {sc:.4f}  {bar}")

    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average : {avg:.4f}")
        print(f"  Runtime : {time.time() - global_start:.1f}s")

    # ====================== REQUIRED STRUCTURED OUTPUT FOR VALIDATOR ======================
    print("\n[STRUCTURED OUTPUT FOR HACKATHON VALIDATOR]", flush=True)

    if scores:
        for task_id, score in scores.items():
            print(f"[START] task={task_id}, [STEP] step=1 reward=0.0, [END] "
                  f"task={task_id} score={score:.4f} steps=1.", flush=True)

        # Overall average block
        avg_score = sum(scores.values()) / len(scores)
        print(f"[START] task=overall, [STEP] step=1 reward=0.0, [END] "
              f"task=overall score={avg_score:.4f} steps=1.", flush=True)

    print("[END OF STRUCTURED OUTPUT]", flush=True)
    # =====================================================================================

if __name__ == "__main__":
    main()
