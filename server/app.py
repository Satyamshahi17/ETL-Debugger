"""
server/app.py
FastAPI server exposing the PipelineEnvironment over HTTP.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

from models import PipelineAction, PipelineObservation
from server.pipeline_environment import PipelineEnvironment

TASK_ID = os.environ.get("TASK_ID", "easy")
env = PipelineEnvironment(task_id=TASK_ID)

app = FastAPI(
    title="ETLDebugger",
    description="OpenEnv-compliant ETL pipeline debugging environment",
    version="1.0.0",
)


class ActionPayload(BaseModel):
    action_type: str = "done"
    column: Optional[str] = None
    params: Dict[str, Any] = {}


@app.get("/")
def root():
    return {
        "name": "ETLDebugger",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.get("/web")
def web():
    return JSONResponse(content={
        "name": "ETLDebugger",
        "description": "OpenEnv-compliant ETL pipeline debugging environment",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    })

@app.post("/reset")
def reset():
    obs = env.reset()
    return JSONResponse(content=_obs_to_dict(obs))


@app.post("/step")
def step(payload: ActionPayload):
    action = PipelineAction(
        action_type=payload.action_type,
        column=payload.column,
        params=payload.params,
    )
    obs = env.step(action)
    return JSONResponse(content=_obs_to_dict(obs))


@app.get("/state")
def state():
    try:
        s = env.state
        return JSONResponse(content={
            "episode_id":         s.episode_id,
            "step_count":         s.step_count,
            "accumulated_reward": s.accumulated_reward,
            "last_action_type":   s.last_action_type,
            "task_id":            s.task_id,
            "is_done":            s.is_done,
        })
    except RuntimeError:
        return JSONResponse(content={"error": "Call /reset first"}, status_code=400)


@app.get("/health")
def health():
    return {"status": "healthy"}


def _obs_to_dict(obs: PipelineObservation) -> dict:
    return {
        "task_id":          obs.task_id,
        "step":             obs.step,
        "dataframe_json":   obs.dataframe_json,
        "column_schema":    obs.column_schema,
        "error_log":        obs.error_log,
        "previous_actions": obs.previous_actions,
        "hint":             obs.hint,
        "done":             obs.done,
        "reward":           obs.reward,
        "metadata":         obs.metadata,
    }