"""
SafeTraj API — FastAPI endpoint for trajectory risk scoring.

Endpoints:
    GET  /              — Health check
    POST /evaluate      — Evaluate trajectory risk for given input commands
    GET  /docs          — Interactive API documentation (auto-generated)

Usage:
    uvicorn api.main:app --reload
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from safetraj import SafeTrajEvaluator, SafeTrajConfig

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="SafeTraj API",
    description=(
        "Trajectory risk scoring API for neural motion prediction models. "
        "Given motion commands (orientation, linear velocity, angular velocity), "
        "returns a risk score, risk label, predicted trajectory, and feature importance."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single evaluator instance ──────────────────────────────
config = SafeTrajConfig(seed=42)
evaluator = SafeTrajEvaluator(config=config)


# ── Request / Response models ──────────────────────────────
class TrajectoryRequest(BaseModel):
    orientation: float = Field(
        default=0.0,
        ge=-3.14,
        le=3.14,
        description="Initial orientation in radians (φ ∈ [-π, π])",
        example=0.0,
    )
    v_lin: float = Field(
        default=0.5,
        ge=-1.05,
        le=2.88,
        description="Linear velocity command in m/s",
        example=0.5,
    )
    v_rot: float = Field(
        default=0.2,
        ge=-1.99,
        le=1.99,
        description="Angular velocity command in rad/s",
        example=0.2,
    )


class TrajectoryResponse(BaseModel):
    orientation: float
    v_lin: float
    v_rot: float
    risk_label: str
    risk_score: float
    mahalanobis_score: float
    isolation_forest_score: float
    feature_importance: dict
    trajectory_length: int


# ── Endpoints ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "SafeTraj API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/evaluate", response_model=TrajectoryResponse, tags=["Evaluation"])
def evaluate(request: TrajectoryRequest):
    """
    Evaluate trajectory risk for given motion commands.

    Returns risk label, risk score, OOD scores, and feature importance.
    """
    x = [request.orientation, request.v_lin, request.v_rot]
    result = evaluator.evaluate(x, return_traj=True)

    traj = result.get("trajectory")
    traj_length = len(traj) if traj is not None else 0

    return TrajectoryResponse(
        orientation=request.orientation,
        v_lin=request.v_lin,
        v_rot=request.v_rot,
        risk_label=result["risk_label"],
        risk_score=round(result["risk_score"], 4),
        mahalanobis_score=round(result["mahalanobis_score"], 4),
        isolation_forest_score=round(result["isolation_forest_score"], 4),
        feature_importance=result["feature_importance"],
        trajectory_length=traj_length,
    )
