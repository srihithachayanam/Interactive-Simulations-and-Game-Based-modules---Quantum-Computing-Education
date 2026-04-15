"""
api.py — Station 07 QML Showdown FastAPI backend

Endpoints:
  GET /api/qml/status
  GET /api/qml/results?sample_size=0.5&noise_level=2&circuit_depth=4
  GET /api/qml/misclassified?sample_size=0.5&noise_level=2&circuit_depth=4

Run:
  uvicorn api:app --reload --port 8000

The frontend should be served from the same origin or CORS is open (dev mode).
"""

import json
import os
import math
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="QML Showdown API", version="1.0.0")

# Dev: allow all origins so the HTML file can call from file:// or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Load precomputed grid ─────────────────────────────────────────────────────
GRID_PATH = os.path.join(os.path.dirname(__file__), "results_grid.json")
_grid: dict = {}

@app.on_event("startup")
async def load_grid():
    global _grid
    if os.path.exists(GRID_PATH):
        with open(GRID_PATH, "r") as f:
            _grid = json.load(f)
        print(f"[API] Loaded {len(_grid)} precomputed configurations.")
    else:
        print("[API] WARNING: results_grid.json not found. Run precompute.py first.")

# ── Valid parameter values ────────────────────────────────────────────────────
VALID_SAMPLE_SIZES   = [0.10, 0.25, 0.50, 0.75, 1.00]
VALID_NOISE_LEVELS   = [0, 1, 2, 3, 4, 5]
VALID_CIRCUIT_DEPTHS = [1, 2, 4, 6, 8]

def snap_to_grid(value: float, valid: list) -> float:
    """Return the closest valid grid value."""
    return min(valid, key=lambda v: abs(v - value))

def grid_key(ss: float, nl: int, cd: int) -> str:
    return f"{ss}_{nl}_{cd}"

def get_result(ss: float, nl: int, cd: int) -> dict:
    key = grid_key(ss, nl, cd)
    if key in _grid:
        return _grid[key]
    # Snap to nearest grid point if exact key missing
    ss2 = snap_to_grid(ss, VALID_SAMPLE_SIZES)
    nl2 = snap_to_grid(nl, VALID_NOISE_LEVELS)
    cd2 = snap_to_grid(cd, VALID_CIRCUIT_DEPTHS)
    key2 = grid_key(ss2, nl2, cd2)
    if key2 in _grid:
        return _grid[key2]
    raise HTTPException(status_code=404, detail=f"No result for {key} or {key2}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/qml/status")
async def status():
    return {
        "precomputed": len(_grid) > 0,
        "grid_size":   len(_grid),
        "expected":    150,
    }

@app.get("/api/qml/results")
async def results(
    sample_size:   float = Query(0.5,  ge=0.0, le=1.0),
    noise_level:   int   = Query(0,    ge=0,   le=5),
    circuit_depth: int   = Query(4,    ge=1,   le=8),
):
    # Snap to grid
    ss = snap_to_grid(sample_size,   VALID_SAMPLE_SIZES)
    nl = snap_to_grid(noise_level,   VALID_NOISE_LEVELS)
    cd = snap_to_grid(circuit_depth, VALID_CIRCUIT_DEPTHS)

    data = get_result(ss, nl, cd)
    return {
        "snapped_to": {"sample_size": ss, "noise_level": nl, "circuit_depth": cd},
        "classical":  data["classical"],
        "quantum":    data["quantum"],
    }

@app.get("/api/qml/misclassified")
async def misclassified(
    sample_size:   float = Query(0.5,  ge=0.0, le=1.0),
    noise_level:   int   = Query(0,    ge=0,   le=5),
    circuit_depth: int   = Query(4,    ge=1,   le=8),
):
    ss = snap_to_grid(sample_size,   VALID_SAMPLE_SIZES)
    nl = snap_to_grid(noise_level,   VALID_NOISE_LEVELS)
    cd = snap_to_grid(circuit_depth, VALID_CIRCUIT_DEPTHS)

    data = get_result(ss, nl, cd)
    return data.get("misclassified", {"classical_only": [], "quantum_only": [], "both": []})

@app.get("/")
async def root():
    return {"message": "QML Showdown API — Station 07", "docs": "/docs"}
