# 🚐 Caravan Layout Optimizer

> **OpenEnv Hackathon — Round 1 Submission**
> A real-world AI environment for optimising caravan interior layouts using iterative LLM inference.

---

## Overview

The **Caravan Layout Optimizer** is an OpenEnv-compatible environment where an AI agent learns to arrange furniture and fixtures inside a caravan to create the best possible floor plan.

The agent places items one at a time onto a **30 × 15 grid** (600 cm × 300 cm, 1 cell = 20 cm) and is evaluated on six real-world criteria:

| Metric | Weight (Hard task) | Description |
|---|---|---|
| **Feasibility** | 0.22 | No overlaps, all items within bounds |
| **Weight Balance** | 0.18 | Left-right and front-back weight distribution |
| **Space Utilisation** | 0.18 | Target ~70% fill rate |
| **Zone Coherence** | 0.16 | Kitchen/dining/sofa front; bed/bathroom rear |
| **Aisle Score** | 0.12 | Columns 13–16 kept clear as walkway |
| **Accessibility** | 0.14 | High-priority items have free adjacent cells |

---

## Why This Environment?

Caravan layout planning is a genuine, hard, real-world optimisation problem:

- **Manufacturers** evaluate thousands of layout permutations to find bestselling configurations
- **Weight regulations** require balanced axle loads for safe towing (legal requirement in most countries)
- **Buyers** need personalised layouts that match their lifestyle (family vs solo traveller)
- **Safety codes** mandate clear egress paths — modelled here as the central aisle

This makes it a rich benchmark: an agent must simultaneously satisfy hard geometric constraints (no overlap, in-bounds) while optimising five competing soft objectives.

---

## Caravan Grid

```
x →  0         13 16       29
     ┌──────────┬────┬──────┐  y=0  (front / hitching end)
     │  FRONT   │    │FRONT │
     │  LEFT    │    │RIGHT │       Kitchen, dining, sofa, fridge
     │          │    │      │
     │──────────┤ISLE│──────│  y=7
     │          │ A  │      │
     │  REAR    │    │REAR  │       Bed, bathroom, wardrobe, storage
     │  LEFT    │    │RIGHT │
     └──────────┴────┴──────┘  y=14 (rear / sleeping end)
```

**1 cell = 20 cm**. Aisle = columns 13–16 (kept free for walking).

---

## Tasks

### 🟢 Task Easy — Basic Placement
Place **3 items** (bed, kitchen unit, storage) without overlaps or boundary violations.

| | |
|---|---|
| Score | `1.0 × feasibility` + 0.1 bonus if all placed |
| Items | `bed_main`, `kitchen_unit`, `storage_a` |
| Max steps | 20 |

### 🟡 Task Medium — Balanced Layout
Place **5 items** optimising feasibility, weight balance, and space utilisation.

| | |
|---|---|
| Score | `0.40 × feasibility + 0.30 × weight_balance + 0.30 × space_utilisation` |
| Items | `bed_main`, `kitchen_unit`, `dining_table`, `bathroom`, `fridge` |
| Max steps | 30 |

### 🔴 Task Hard — Full Caravan Design
Place **all 9 items** with full multi-objective scoring including zone coherence, aisle preservation, and accessibility.

| | |
|---|---|
| Score | `0.22 × feasibility + 0.18 × weight_balance + 0.18 × space_utilisation + 0.16 × zone_coherence + 0.12 × aisle_score + 0.14 × accessibility` |
| Items | All 9 (see catalogue below) |
| Max steps | 60 |

---

## Items Catalogue

| Item ID | Size (cells) | Size (cm) | Weight | Zone | Priority |
|---|---|---|---|---|---|
| `bed_main` | 10 × 5 | 200 × 100 cm | 40 kg | Rear | 1 (high) |
| `kitchen_unit` | 6 × 3 | 120 × 60 cm | 30 kg | Front | 1 (high) |
| `dining_table` | 5 × 4 | 100 × 80 cm | 15 kg | Front | 2 |
| `storage_a` | 4 × 3 | 80 × 60 cm | 20 kg | Any | 3 (low) |
| `storage_b` | 4 × 3 | 80 × 60 cm | 20 kg | Any | 3 (low) |
| `bathroom` | 5 × 5 | 100 × 100 cm | 50 kg | Rear | 1 (high) |
| `sofa` | 7 × 3 | 140 × 60 cm | 25 kg | Front | 2 |
| `wardrobe` | 4 × 2 | 80 × 40 cm | 18 kg | Rear | 3 (low) |
| `fridge` | 2 × 3 | 40 × 60 cm | 22 kg | Front | 1 (high) |

Items can be rotated 90° (swaps width and height).

---

## API Reference

All endpoints conform to the **OpenEnv specification** (`openenv.yaml`).

### `GET /` — Health Check
```json
{ "status": "ok", "environment": "CaravanLayoutOptimizer", "version": "1.0.0" }
```

### `GET /tasks` — List Tasks
Returns all 3 task objects with id, name, difficulty, description, scoring formula.

### `GET /items` — Item Catalogue
Returns all items with dimensions, weight, zone preference.

### `POST /reset` — Reset Environment
```json
{ "task_id": "task_easy" }
```
Body is optional — defaults to `task_easy`. Returns initial `EnvironmentState`.

### `POST /step` — Place One Item
```json
{
  "item_id": "bed_main",
  "x": 17,
  "y": 9,
  "rotation": 90
}
```
Returns `StepResult`: updated state, reward, done flag, info dict.

### `GET /state` — Current State
Returns full `EnvironmentState` without advancing the episode.

### `GET /grid` — ASCII Visualisation
Returns the current layout as a human-readable grid + live metrics.

---

## Observation Space

```python
EnvironmentState:
  task_id          str
  grid_width       int               # 30
  grid_height      int               # 15
  placed_items     List[PlacedItem]  # items already on the grid
  unplaced_items   List[CaravanItem] # items still to place
  step_count       int
  done             bool
  score            float             # [0.0, 1.0]
  metrics:
    feasibility        float
    weight_balance     float
    space_utilisation  float
    zone_coherence     float
    aisle_score        float
    accessibility      float
    items_placed       float
    items_remaining    float
  grid_snapshot    List[List[str]]   # 2D visual grid
```

## Action Space

```python
StepAction:
  item_id   str          # must be in unplaced_items
  x         int [0–29]   # column, 0 = left wall
  y         int [0–14]   # row, 0 = front of caravan
  rotation  int {0, 90}  # degrees; 90 swaps width/height
```

---

## Reward Design

| Event | Reward |
|---|---|
| Valid placement | `+0.05` |
| Valid + correct zone | `+0.07` |
| Invalid placement (overlap / out of bounds) | `−0.10` |
| Unknown item ID | `−0.05` |
| Episode end | `+final_score` (terminal, grader result) |

Dense intermediate rewards guide the agent toward valid, zone-aware placements. The terminal reward from the grader provides the true multi-objective signal.

---

## Inference Architecture — Iterative Step-by-Step

Unlike a naive plan-then-execute approach, `inference.py` implements a **true feedback loop**:

```
WHILE items remain:
  1. Read LIVE state from env (after every placement)
  2. Select next item (highest accessibility priority first)
  3. Ask LLM: "place THIS ONE item" with full context:
       - Exact occupied cell ranges of every placed item
       - Free cell count per quadrant
       - Current score and all 6 metrics
       - Items still to place after this one
  4. Client-side pre-validate: bounds + overlap check before calling env
  5. Execute step() in environment
  6. If REJECTED → inject exact error into conversation → LLM retries (up to 3×)
  7. If all retries fail → heuristic scan fallback for that item only
  8. If ACCEPTED → inject success + reward into conversation history
  9. Loop to next item
```

Key properties:
- **Multi-turn conversation history** — LLM remembers every prior placement
- **Per-item retry with error injection** — LLM corrects based on exact rejection reason
- **Client-side pre-validation** — catches bad coordinates before wasting env steps
- **Per-item heuristic fallback** — one bad LLM response never breaks the whole layout
- **Priority-first ordering** — high-access items (bed, kitchen, bathroom) get best spots first

---

## Project Structure

```
caravan-layout-optimizer/
├── main.py                  # FastAPI server (OpenEnv endpoints)
├── inference.py             # Iterative LLM baseline
├── openenv.yaml             # OpenEnv specification
├── Dockerfile               # HuggingFace Spaces / Docker
├── requirements.txt
├── README.md
└── env/
    ├── __init__.py
    ├── models.py            # Pydantic typed models
    ├── tasks.py             # Task definitions & item catalogue
    ├── graders.py           # 6 scoring functions + 3 task graders
    └── caravan_env.py       # Core state machine (reset/step/state)
```

---

## Setup & Running

### Local Development

```bash
git clone <your-repo-url>
cd caravan-layout-optimizer
pip install -r requirements.txt

# Start the environment server
uvicorn main:app --host 0.0.0.0 --port 7860

# Verify health
curl http://localhost:7860/

# Run iterative inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### Docker

```bash
docker build -t caravan-optimizer .

docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  caravan-optimizer
```

### Hugging Face Spaces

1. Create a new Space → **Docker** SDK
2. Connect your GitHub repo
3. Add Secrets (Settings → Repository Secrets):
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. The server starts automatically on port 7860

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ | LLM API base URL (OpenAI-compatible) |
| `MODEL_NAME` | ✅ | Model identifier, e.g. `gpt-4o-mini` |
| `HF_TOKEN` | ✅ | API key used for LLM calls |
| `ENV_BASE_URL` | ⬜ optional | Override env server URL (default: `http://localhost:7860`) |

---

## Pre-Submission Checklist

- [x] `GET /` returns 200 with `status: ok`
- [x] `POST /reset` works with and without request body
- [x] `POST /step` validates action and returns reward + state
- [x] `GET /state` returns current state without side effects
- [x] All 3 tasks defined with graders returning scores in `[0.0, 1.0]`
- [x] `openenv.yaml` spec matches actual API and metrics
- [x] `inference.py` named correctly, placed in root directory
- [x] Inference uses `OpenAI` client with `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN`
- [x] Inference runtime < 20 min (typically ~5 min for all 3 tasks)
- [x] Dockerfile builds and exposes port 7860
- [x] Runs within 2 vCPU / 8 GB RAM constraint
