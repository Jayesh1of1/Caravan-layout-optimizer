from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class Rotation(int, Enum):
    DEG_0 = 0
    DEG_90 = 90


class ItemType(str, Enum):
    BED = "bed"
    KITCHEN = "kitchen"
    DINING_TABLE = "dining_table"
    STORAGE = "storage"
    BATHROOM = "bathroom"
    SOFA = "sofa"
    WARDROBE = "wardrobe"
    FRIDGE = "fridge"


class CaravanItem(BaseModel):
    id: str
    item_type: ItemType
    # dimensions in grid cells (1 cell = 20cm)
    width: int
    height: int
    weight_kg: float
    zone_preference: str  # "front", "rear", "any"
    accessibility_priority: int  # 1=high, 2=medium, 3=low

    def rotated_dims(self, rotation: int):
        if rotation == 90:
            return self.height, self.width
        return self.width, self.height


class PlacedItem(BaseModel):
    item: CaravanItem
    x: int  # grid column (left-right)
    y: int  # grid row (front-back)
    rotation: int = 0  # 0 or 90

    @property
    def effective_width(self):
        return self.item.height if self.rotation == 90 else self.item.width

    @property
    def effective_height(self):
        return self.item.width if self.rotation == 90 else self.item.height

    def cells(self):
        """Returns set of (x, y) grid cells this item occupies."""
        cells = set()
        for dx in range(self.effective_width):
            for dy in range(self.effective_height):
                cells.add((self.x + dx, self.y + dy))
        return cells


class StepAction(BaseModel):
    item_id: str = Field(..., description="ID of the item to place")
    x: int = Field(..., description="Grid X position (column, 0=left wall)")
    y: int = Field(..., description="Grid Y position (row, 0=front)")
    rotation: int = Field(default=0, description="Rotation in degrees: 0 or 90")


class EnvironmentState(BaseModel):
    task_id: str
    grid_width: int
    grid_height: int
    placed_items: List[PlacedItem]
    unplaced_items: List[CaravanItem]
    step_count: int
    done: bool
    score: float
    metrics: Dict[str, float]
    grid_snapshot: List[List[str]]  # visual grid


class StepResult(BaseModel):
    state: EnvironmentState
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str = Field(default="task_easy", description="One of: task_easy, task_medium, task_hard")


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    items_to_place: List[str]
