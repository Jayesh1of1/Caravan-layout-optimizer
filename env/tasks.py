from env.models import CaravanItem, ItemType, TaskInfo

# ---------------------------------------------------------------------------
# Caravan grid: 30 cols (x) × 15 rows (y)  →  600 cm × 300 cm (6 m × 3 m)
# 1 cell = 20 cm
# ---------------------------------------------------------------------------
GRID_WIDTH = 30   # left-right (600 cm)
GRID_HEIGHT = 15  # front-back (300 cm)

# ---------------------------------------------------------------------------
# Item catalogue
# ---------------------------------------------------------------------------
ITEM_CATALOGUE: dict[str, CaravanItem] = {
    "bed_main": CaravanItem(
        id="bed_main", item_type=ItemType.BED,
        width=10, height=5, weight_kg=40.0,
        zone_preference="rear", accessibility_priority=1
    ),
    "kitchen_unit": CaravanItem(
        id="kitchen_unit", item_type=ItemType.KITCHEN,
        width=6, height=3, weight_kg=30.0,
        zone_preference="front", accessibility_priority=1
    ),
    "dining_table": CaravanItem(
        id="dining_table", item_type=ItemType.DINING_TABLE,
        width=5, height=4, weight_kg=15.0,
        zone_preference="front", accessibility_priority=2
    ),
    "storage_a": CaravanItem(
        id="storage_a", item_type=ItemType.STORAGE,
        width=4, height=3, weight_kg=20.0,
        zone_preference="any", accessibility_priority=3
    ),
    "storage_b": CaravanItem(
        id="storage_b", item_type=ItemType.STORAGE,
        width=4, height=3, weight_kg=20.0,
        zone_preference="any", accessibility_priority=3
    ),
    "bathroom": CaravanItem(
        id="bathroom", item_type=ItemType.BATHROOM,
        width=5, height=5, weight_kg=50.0,
        zone_preference="rear", accessibility_priority=1
    ),
    "sofa": CaravanItem(
        id="sofa", item_type=ItemType.SOFA,
        width=7, height=3, weight_kg=25.0,
        zone_preference="front", accessibility_priority=2
    ),
    "wardrobe": CaravanItem(
        id="wardrobe", item_type=ItemType.WARDROBE,
        width=4, height=2, weight_kg=18.0,
        zone_preference="rear", accessibility_priority=3
    ),
    "fridge": CaravanItem(
        id="fridge", item_type=ItemType.FRIDGE,
        width=2, height=3, weight_kg=22.0,
        zone_preference="front", accessibility_priority=1
    ),
    "bed_bunk": CaravanItem(
        id="bed_bunk", item_type=ItemType.BED,
        width=8, height=4, weight_kg=35.0,
        zone_preference="rear", accessibility_priority=2
    ),
}

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: dict[str, TaskInfo] = {
    "task_easy": TaskInfo(
        id="task_easy",
        name="Basic Placement",
        description=(
            "Place 3 essential items (bed, kitchen, storage) inside the caravan "
            "without any overlaps and within the grid bounds. "
            "Score is based purely on feasibility: all items placed without collision."
        ),
        difficulty="easy",
        max_steps=20,
        items_to_place=["bed_main", "kitchen_unit", "storage_a"],
    ),
    "task_medium": TaskInfo(
        id="task_medium",
        name="Balanced Layout",
        description=(
            "Place 5 items (bed, kitchen, dining table, bathroom, fridge) without overlap, "
            "and optimise for left-right weight balance and space utilisation. "
            "Score = 0.4×feasibility + 0.3×weight_balance + 0.3×space_utilisation."
        ),
        difficulty="medium",
        max_steps=30,
        items_to_place=["bed_main", "kitchen_unit", "dining_table", "bathroom", "fridge"],
    ),
    "task_hard": TaskInfo(
        id="task_hard",
        name="Full Caravan Design",
        description=(
            "Place all 9 items without overlap, optimising for weight balance, "
            "space utilisation, zone coherence (kitchen/dining front, bed/bathroom rear), "
            "a clear central aisle (columns 13–16 mostly free), and priority accessibility. "
            "Score = 0.22×feasibility + 0.18×weight_balance + 0.18×space_utilisation "
            "+ 0.16×zone_coherence + 0.12×aisle_score + 0.14×accessibility."
        ),
        difficulty="hard",
        max_steps=60,
        items_to_place=[
            "bed_main", "kitchen_unit", "dining_table",
            "storage_a", "storage_b", "bathroom",
            "sofa", "wardrobe", "fridge"
        ],
    ),
}
