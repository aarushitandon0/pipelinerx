"""FastAPI application — all 5 HTTP endpoints for the PipelineRx environment."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI

from app.environment import env_state
from app.models import (
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TaskInfo,
    InfoDict,
)
from app.tasks import TASK_INFOS

app = FastAPI(
    title="PipelineRx",
    description=(
        "An RL environment where AI agents diagnose and repair silent, "
        "real-world ML pipeline failures."
    ),
    version="1.0.0",
)


# ── Eagerly initialise so /health never races ──────────────────────────────
@app.on_event("startup")
def _startup():
    """Pre-load task generators so cold-start latency is minimal."""
    from app.tasks import GENERATORS  # noqa: F401 — import triggers module load


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check. Returns {"status": "ok"} with HTTP 200."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    """Start a new episode. Generates a fresh broken DataFrame."""
    result = env_state.reset(task_id=request.task_id, seed=request.seed)
    return ResetResponse(
        task_id=result["task_id"],
        observation=result["observation"],
        info=InfoDict(**result["info"]),
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute one action. Returns observation, reward, done, info.

    Always returns HTTP 200 — never 4xx to the agent.
    """
    result = env_state.step(action=request.action, parameters=request.parameters)
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"],
        info=InfoDict(**result["info"]),
    )


@app.get("/state", response_model=StateResponse)
def state():
    """Return current environment state."""
    s = env_state.get_state()
    return StateResponse(**s)


@app.get("/tasks", response_model=List[TaskInfo])
def tasks():
    """List all 5 tasks with metadata."""
    return [TaskInfo(**info) for info in TASK_INFOS.values()]
