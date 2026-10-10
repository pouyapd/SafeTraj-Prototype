"""
SafeTraj API — REST access to the same evaluator used by the CLI and dashboard.

Endpoints:
    GET  /                     service info
    GET  /health               health check
    GET  /config               thresholds, operational ranges and predictors in use
    POST /evaluate             one command (orientation, v_lin, v_rot) toward a goal
    POST /evaluate/trajectory  a trajectory you supply (e.g. your model's output)
    POST /evaluate/batch       up to 500 commands
    POST /compare              two commands toward the same goal
    POST /report               plain-text report for a command (template; no LLM on the server)
    GET  /docs                 interactive documentation (generated)

Usage:
    uvicorn api.main:app --reload

Scores are heuristics: "checks" are fixed-threshold rules and "novelty" says
how unusual the input is compared with reference commands. Neither is a
calibrated probability of a collision or a safety certificate.
"""

import math
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from safetraj import SafeTrajConfig, SafeTrajEvaluator, Trajectory, TrajectoryError, __version__  # noqa: E402
from safetraj.config import OPERATIONAL_RANGES  # noqa: E402
from safetraj.report import template_report  # noqa: E402

app = FastAPI(
    title="SafeTraj API",
    description=(
        "Trajectory-level checks and input-novelty scores for trajectory predictors. "
        "Checks are fixed-threshold heuristics and novelty compares an input with reference commands; "
        "neither is a calibrated probability of a collision."
    ),
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CONFIG = SafeTrajConfig(seed=42)
PredictorName = Literal["goal_seeking", "constant_command"]
MAX_BATCH = 500


@lru_cache(maxsize=4)
def get_evaluator(predictor: str) -> SafeTrajEvaluator:
    return SafeTrajEvaluator(CONFIG, predictor=predictor)


# ── Request models ──────────────────────────────────────────────────────────

O_LO, O_HI = OPERATIONAL_RANGES["orientation"]
V_LO, V_HI = OPERATIONAL_RANGES["v_lin"]
W_LO, W_HI = OPERATIONAL_RANGES["v_rot"]


class CommandRequest(BaseModel):
    orientation: float = Field(0.0, ge=O_LO, le=O_HI, description="Initial orientation θ in radians, [-π, π]")
    v_lin: float = Field(0.5, ge=V_LO, le=V_HI, description="Linear velocity in m/s, [-1.05, 2.88]")
    v_rot: float = Field(0.2, ge=W_LO, le=W_HI, description="Angular velocity in rad/s, [-1.99, 1.99]")
    goal_x: float = Field(1.0, ge=-10, le=10, description="Goal x in metres (robot starts at the origin)")
    goal_y: float = Field(0.0, ge=-10, le=10, description="Goal y in metres")
    predictor: PredictorName = Field("goal_seeking", description="Built-in reference predictor")
    include_trajectory: bool = Field(False, description="Return the predicted trajectory points")

    model_config = {"json_schema_extra": {"examples": [
        {"orientation": 0.0, "v_lin": 0.5, "v_rot": 0.2, "goal_x": 1.0, "goal_y": 0.0}]}}


class Point(BaseModel):
    t: float
    x: float
    y: float
    theta: float
    v: float
    omega: float

    @field_validator("*")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class TrajectoryRequest(BaseModel):
    points: List[Point] = Field(..., min_length=2, max_length=10_000)
    goal_x: float
    goal_y: float
    traj_id: str = "trajectory"
    command: Optional[List[float]] = Field(None, min_length=3, max_length=3,
                                           description="Initial [orientation, v_lin, v_rot] for the novelty score")


class BatchRequest(BaseModel):
    commands: List[List[float]] = Field(..., min_length=1, description=f"Up to {MAX_BATCH} [orientation, v_lin, v_rot]")
    goal_x: float = Field(1.0, ge=-10, le=10)
    goal_y: float = Field(0.0, ge=-10, le=10)
    predictor: PredictorName = "goal_seeking"

    @field_validator("commands")
    @classmethod
    def check_commands(cls, cmds: List[List[float]]) -> List[List[float]]:
        if len(cmds) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} commands per request")
        ranges = (OPERATIONAL_RANGES["orientation"], OPERATIONAL_RANGES["v_lin"], OPERATIONAL_RANGES["v_rot"])
        for i, cmd in enumerate(cmds):
            if len(cmd) != 3:
                raise ValueError(f"command {i} must have 3 values [orientation, v_lin, v_rot]")
            for value, (lo, hi), name in zip(cmd, ranges, ("orientation", "v_lin", "v_rot")):
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"command {i}: {name}={value} outside [{lo:g}, {hi:g}]")
        return cmds


class CompareRequest(BaseModel):
    a: CommandRequest
    b: CommandRequest


# ── Helpers ─────────────────────────────────────────────────────────────────

def _public(result: Dict[str, Any], include_trajectory: bool = False) -> Dict[str, Any]:
    out = {k: v for k, v in result.items()
           if k not in ("trajectory", "mahalanobis_score", "isolation_forest_score", "risk_score", "risk_label")}
    if include_trajectory and "trajectory" in result:
        out["trajectory"] = result["trajectory"].to_dict()["points"]
    return out


@app.exception_handler(TrajectoryError)
async def trajectory_error_handler(request: Request, exc: TrajectoryError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {"service": "SafeTraj API", "version": __version__, "docs": "/docs",
            "endpoints": ["/evaluate", "/evaluate/trajectory", "/evaluate/batch", "/compare", "/report", "/config"]}


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok"}


@app.get("/config", tags=["Info"])
def config():
    c = CONFIG
    return {
        "operational_ranges": OPERATIONAL_RANGES,
        "goal_thresholds_m": {"strict_success": c.strict_threshold, "near_success": c.near_threshold},
        "rule_thresholds": {"progress_min_m": c.rule_progress_min, "omega_max": c.rule_omega_max,
                            "domega_max_per_sample": c.rule_domega_max, "heading_error_far_rad": c.rule_heading_far},
        "novelty_percentiles": {"borderline": c.borderline_percentile, "novel": c.novel_percentile},
        "reference_data": f"{c.n_train_samples} synthetic commands, uniform over the operational ranges (seed {c.seed})",
        "prediction": {"dt_s": c.dt, "steps": c.num_steps, "predictors": ["goal_seeking", "constant_command"]},
    }


@app.post("/evaluate", tags=["Evaluation"])
def evaluate(req: CommandRequest):
    """Predict, measure, check and score one command."""
    ev = get_evaluator(req.predictor)
    res = ev.evaluate([req.orientation, req.v_lin, req.v_rot], return_traj=True, goal=(req.goal_x, req.goal_y))
    return _public(res, req.include_trajectory)


@app.post("/evaluate/trajectory", tags=["Evaluation"])
def evaluate_trajectory(req: TrajectoryRequest):
    """Measure and check a trajectory you provide; novelty only if `command` is given."""
    if req.command is not None and not all(math.isfinite(v) for v in req.command):
        raise HTTPException(status_code=422, detail="command values must be finite numbers")
    if not (math.isfinite(req.goal_x) and math.isfinite(req.goal_y)):
        raise HTTPException(status_code=422, detail="goal must be finite")
    traj = Trajectory.from_points([p.model_dump() for p in req.points], (req.goal_x, req.goal_y),
                                  traj_id=req.traj_id, command=req.command)
    res = get_evaluator("goal_seeking").evaluate_trajectory(traj)
    res["predictor"] = "user"
    return _public(res)


@app.post("/evaluate/batch", tags=["Evaluation"])
def evaluate_batch(req: BatchRequest):
    ev = get_evaluator(req.predictor)
    results = [_public(r) for r in ev.evaluate_commands(req.commands, goal=(req.goal_x, req.goal_y))]
    counts = {lab: sum(r["checks"]["label"] == lab for r in results) for lab in ("PASS", "REVIEW", "FAIL")}
    return {"n": len(results), "check_labels": counts, "results": results}


@app.post("/compare", tags=["Evaluation"])
def compare(req: CompareRequest):
    if (req.a.goal_x, req.a.goal_y) != (req.b.goal_x, req.b.goal_y) or req.a.predictor != req.b.predictor:
        raise HTTPException(status_code=422, detail="a and b must use the same goal and predictor")
    ev = get_evaluator(req.a.predictor)
    out = ev.compare([req.a.orientation, req.a.v_lin, req.a.v_rot], [req.b.orientation, req.b.v_lin, req.b.v_rot],
                     goal=(req.a.goal_x, req.a.goal_y))
    return {"a": _public(out["a"], req.a.include_trajectory), "b": _public(out["b"], req.b.include_trajectory),
            "difference_b_minus_a": out["difference_b_minus_a"]}


@app.post("/report", tags=["Evaluation"])
def report(req: CommandRequest):
    """Deterministic text report (the server does not call an LLM)."""
    res = get_evaluator(req.predictor).evaluate([req.orientation, req.v_lin, req.v_rot],
                                                goal=(req.goal_x, req.goal_y))
    return {"report": template_report(res), "checks": res["checks"]["label"], "novelty": res["novelty"]["label"]}
