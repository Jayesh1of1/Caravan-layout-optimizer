"""
Core CaravanEnv — implements the OpenEnv step()/reset()/state() contract.
"""
from __future__ import annotations
import copy
from typing import List, Dict, Any

from env.models import (
    CaravanItem, PlacedItem, StepAction, EnvironmentState,
    StepResult, ResetRequest
)
from env.tasks import TASKS, ITEM_CATALOGUE, GRID_WIDTH, GRID_HEIGHT
from env.graders import GRADERS


class CaravanEnv:
    """Stateful caravan layout optimisation environment."""

    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self._task_id: str = "task_easy"
        self._placed_items: List[PlacedItem] = []
        self._unplaced_items: List[CaravanItem] = []
        self._step_count: int = 0
        self._done: bool = False
        self._score: float = 0.0
        self._max_steps: int = 20
        self.total_items: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def placed_items(self) -> List[PlacedItem]:
        return self._placed_items

    @property
    def task_id(self) -> str:
        return self._task_id

    # ------------------------------------------------------------------
    # reset() (FIXED — NO CRASH)
    # ------------------------------------------------------------------
    def reset(self, request: ResetRequest) -> EnvironmentState:
        task_id = getattr(request, "task_id", "task_easy") or "task_easy"

        if task_id not in TASKS:
            task_id = "task_easy"

        task = TASKS[task_id]

        self._task_id = task_id
        self._max_steps = task.max_steps
        self._step_count = 0
        self._done = False
        self._score = 0.0

        self._unplaced_items = [
            copy.deepcopy(ITEM_CATALOGUE[iid]) for iid in task.items_to_place
        ]
        self._placed_items = []
        self.total_items = len(self._unplaced_items)

        return self._build_state()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, action: StepAction) -> StepResult:
        if self._done:
            state = self._build_state()
            return StepResult(
                state=state, reward=0.0, done=True,
                info={"error": "Episode already done. Call reset()."}
            )

        info: Dict[str, Any] = {}

        item_idx = next(
            (i for i, it in enumerate(self._unplaced_items) if it.id == action.item_id),
            None
        )

        if item_idx is None:
            already = any(pi.item.id == action.item_id for pi in self._placed_items)
            info["warning"] = (
                f"Item '{action.item_id}' already placed."
                if already else f"Item '{action.item_id}' not found."
            )
            reward = -0.05
        else:
            item = self._unplaced_items[item_idx]
            placed = PlacedItem(item=item, x=action.x, y=action.y, rotation=action.rotation)

            valid, reason = self._validate_placement(placed)

            if valid:
                self._unplaced_items.pop(item_idx)
                self._placed_items.append(placed)
                info["placed"] = action.item_id
                reward = self._incremental_reward(placed)
            else:
                info["invalid_reason"] = reason
                reward = -0.1

        self._step_count += 1

        all_placed = len(self._unplaced_items) == 0
        out_of_steps = self._step_count >= self._max_steps

        if all_placed or out_of_steps:
            self._done = True
            self._score = self._compute_final_score()
            reward += self._score
            info["final_score"] = self._score

        state = self._build_state()
        return StepResult(state=state, reward=reward, done=self._done, info=info)

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------
    def state(self) -> EnvironmentState:
        return self._build_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_placement(self, placed: PlacedItem):
        if placed.x < 0 or placed.y < 0:
            return False, "Negative coordinates"

        if placed.x + placed.effective_width > self.grid_width:
            return False, "Exceeds grid width"

        if placed.y + placed.effective_height > self.grid_height:
            return False, "Exceeds grid height"

        new_cells = placed.cells()

        for existing in self._placed_items:
            if new_cells & existing.cells():
                return False, f"Overlaps with '{existing.item.id}'"

        return True, "OK"

    def _incremental_reward(self, placed: PlacedItem) -> float:
        reward = 0.05
        mid_y = self.grid_height / 2
        item_cy = placed.y + placed.effective_height / 2
        pref = placed.item.zone_preference

        if pref == "front" and item_cy < mid_y:
            reward += 0.02
        elif pref == "rear" and item_cy >= mid_y:
            reward += 0.02

        return reward

    def _compute_final_score(self) -> float:
        grader = GRADERS.get(self._task_id)
        if grader is None:
            return 0.0
        return round(max(0.0, min(1.0, grader(self))), 4)

    def _build_state(self) -> EnvironmentState:
        score = self._score if self._done else self._compute_final_score()
        metrics = self._compute_metrics()
        grid = self._render_grid()

        return EnvironmentState(
            task_id=self._task_id,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            placed_items=list(self._placed_items),
            unplaced_items=list(self._unplaced_items),
            step_count=self._step_count,
            done=self._done,
            score=score,
            metrics=metrics,
            grid_snapshot=grid,
        )

    # ------------------------------------------------------------------
    # METRICS (UPGRADED — 10/10)
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> Dict[str, float]:
        from env.graders import (
            _feasibility_score,
            _weight_balance_score,
            _space_utilisation_score,
            _zone_coherence_score,
            _aisle_score,
            _priority_accessibility_score,
        )

        return {
            "feasibility": round(_feasibility_score(self), 4),
            "weight_balance": round(_weight_balance_score(self), 4),
            "space_utilisation": round(_space_utilisation_score(self), 4),
            "zone_coherence": round(_zone_coherence_score(self), 4),
            "aisle_score": round(_aisle_score(self), 4),
            "accessibility": round(_priority_accessibility_score(self), 4),
            "items_placed": float(len(self._placed_items)),
            "items_remaining": float(len(self._unplaced_items)),
        }

    # ------------------------------------------------------------------
    # GRID RENDER
    # ------------------------------------------------------------------
    def _render_grid(self) -> List[List[str]]:
        grid = [["." for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        abbreviations = {
            "bed_main": "BM",
            "kitchen_unit": "KU",
            "dining_table": "DT",
            "storage_a": "SA",
            "storage_b": "SB",
            "bathroom": "BA",
            "sofa": "SF",
            "wardrobe": "WD",
            "fridge": "FR",
            "bed_bunk": "BB",
        }

        for pi in self._placed_items:
            label = abbreviations.get(pi.item.id, pi.item.id[:2].upper())

            for (cx, cy) in pi.cells():
                if 0 <= cy < self.grid_height and 0 <= cx < self.grid_width:
                    grid[cy][cx] = label

        return grid