"""Pydantic request/response models following the OpenEnv specification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=5, description="Which task to start (1-5)")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: str = Field(..., description="Action name, e.g. inspect_data, apply_fix")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


# ── Response models ─────────────────────────────────────────────────────────

class InfoDict(BaseModel):
    tests_passed: int = 0
    tests_total: int = 5
    step_count: int = 0
    task_id: int = 1
    max_steps: int = 30
    warnings: List[str] = Field(default_factory=list)
    root_cause_stage: Optional[int] = None
    # Optional session token returned by /reset. Agents may include this
    # token in subsequent StepRequest.parameters as `session_id` to protect
    # against stale-step races. Not required for compatibility.
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    task_id: int
    observation: str
    info: InfoDict


class StepResponse(BaseModel):
    observation: str
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: InfoDict


class StateResponse(BaseModel):
    task_id: int
    step_count: int
    reward: float
    df_shape: List[int]
    column_types: Dict[str, str]
    max_steps: int
    done: bool


class TaskInfo(BaseModel):
    id: int
    name: str
    difficulty: str
    description: str
    max_steps: int
    hint: str


class HealthResponse(BaseModel):
    status: str = "ok"
