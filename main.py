"""
Caravan Layout Optimizer — OpenEnv-compatible FastAPI server.
Endpoints: GET /  GET /tasks  GET /state  POST /reset  POST /step
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from fastapi import Body
import uvicorn

from env.caravan_env import CaravanEnv
from env.models import (
    StepAction, StepResult, EnvironmentState,
    ResetRequest, TaskInfo,
)
from env.tasks import TASKS, ITEM_CATALOGUE, GRID_WIDTH, GRID_HEIGHT

app = FastAPI(
    title="Caravan Layout Optimizer",
    description=(
        "An OpenEnv-compatible environment where an AI agent learns to arrange "
        "furniture and fixtures inside a caravan to maximise space efficiency, "
        "weight balance, zone coherence, and accessibility."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session; for production use session keys)
env = CaravanEnv()


@app.get("/", summary="Health check")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "environment": "CaravanLayoutOptimizer",
        "version": "1.0.0",
        "grid": f"{GRID_WIDTH}×{GRID_HEIGHT} cells (600 cm × 300 cm)",
        "tasks": list(TASKS.keys()),
    }


@app.get("/tasks", response_model=List[TaskInfo], summary="List all tasks")
def list_tasks():
    """Returns all available tasks with their descriptions and difficulty."""
    return list(TASKS.values())


@app.get("/items", summary="List all caravan items in the catalogue")
def list_items() -> Dict[str, Any]:
    return {
        iid: {
            "id": item.id,
            "type": item.item_type,
            "width_cells": item.width,
            "height_cells": item.height,
            "width_cm": item.width * 20,
            "height_cm": item.height * 20,
            "weight_kg": item.weight_kg,
            "zone_preference": item.zone_preference,
        }
        for iid, item in ITEM_CATALOGUE.items()
    }


@app.post("/reset", response_model=EnvironmentState, summary="Reset environment")
def reset(request: Optional[ResetRequest] = Body(default=None)):
    """
    Reset the environment for a given task.
    task_id: one of 'task_easy', 'task_medium', 'task_hard'
    Body is optional — defaults to task_easy if omitted.
    """
    if request is None:
        request = ResetRequest(task_id="task_easy")
    try:
        state = env.reset(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state


@app.post("/step", response_model=StepResult, summary="Place one item")
def step(action: StepAction):
    """
    Place an item on the caravan grid.
    - item_id: e.g. 'bed_main', 'kitchen_unit'
    - x, y: top-left grid coordinates (0-indexed)
    - rotation: 0 or 90 (degrees)
    """
    result = env.step(action)
    return result


@app.get("/state", response_model=EnvironmentState, summary="Get current state")
def state():
    """Returns the current environment state without advancing it."""
    return env.state()


@app.get("/grid", summary="Render current grid as ASCII")
def render_grid() -> Dict[str, Any]:
    """Returns the current layout as a human-readable ASCII grid."""
    s = env.state()
    rows = ["  " + "".join(f"{x:2d}" for x in range(s.grid_width))]
    for y, row in enumerate(s.grid_snapshot):
        rows.append(f"{y:2d} " + " ".join(row))
    return {
        "grid_ascii": "\n".join(rows),
        "legend": {
            "BM": "Bed Main", "KU": "Kitchen Unit", "DT": "Dining Table",
            "SA": "Storage A", "SB": "Storage B", "BA": "Bathroom",
            "SF": "Sofa", "WD": "Wardrobe", "FR": "Fridge", ".": "Empty",
        },
        "metrics": s.metrics,
        "score": s.score,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
