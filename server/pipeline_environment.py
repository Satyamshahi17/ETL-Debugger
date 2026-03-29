"""
server/pipeline_environment.py
OpenEnv-compliant Environment per official docs pattern.
Extends openenv.core.env_server.interfaces.Environment.
State is used directly from openenv (not subclassed).
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import PipelineAction, PipelineObservation, PipelineState
from tasks.fixtures import load_task
from server.actions import apply_action
from server.reward import compute_reward, compute_reward_breakdown

MAX_STEPS = 20
HINT_UNLOCK_AFTER = 5

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State as OpenEnvState
    _OPENENV = True
except ImportError:
    _OPENENV = False
    class Environment:
        pass


class PipelineEnvironment(Environment):

    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self._fixture = load_task(task_id)
        self._df: Optional[pd.DataFrame] = None
        self._pipeline_state: Optional[PipelineState] = None
        self._history: list = []

    def reset(self) -> PipelineObservation:
        self._df = self._fixture.broken_df.copy()
        self._pipeline_state = PipelineState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            accumulated_reward=0.0,
            last_action_type="",
            task_id=self.task_id,
            is_done=False,
        )
        self._history = []
        return self._build_obs(error_log=[], reward=0.0, done=False)

    def step(self, action: PipelineAction) -> PipelineObservation:
        if self._pipeline_state is None or self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._pipeline_state.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._pipeline_state.step_count += 1
        errors: list[str] = []

        if action.action_type != "done":
            try:
                self._df = apply_action(self._df, action)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        self._pipeline_state.last_action_type = action.action_type
        self._history.append(action.action_type)
        history = self._history

        reward = compute_reward(self._df, self._fixture.ground_truth_df, history)
        breakdown = compute_reward_breakdown(self._df, self._fixture.ground_truth_df, history)
        self._pipeline_state.accumulated_reward += reward

        done = (action.action_type == "done" or self._pipeline_state.step_count >= MAX_STEPS)
        self._pipeline_state.is_done = done

        hint = self._maybe_hint(reward)
        return self._build_obs(error_log=errors, reward=reward, done=done,
                               hint=hint, metadata=breakdown)

    @property
    def state(self) -> PipelineState:
        if self._pipeline_state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._pipeline_state

    def _build_obs(self, error_log, reward, done, hint=None, metadata=None) -> PipelineObservation:
        df = self._df if self._df is not None else self._fixture.broken_df
        schema = [
            {
                "name":     col,
                "dtype":    str(df[col].dtype),
                "nullable": bool(df[col].isnull().any()),
                "n_nulls":  int(df[col].isnull().sum()),
                "sample":   df[col].dropna().head(3).tolist(),
            }
            for col in df.columns
        ]
        return PipelineObservation(
            task_id=self.task_id,
            step=self._pipeline_state.step_count if self._pipeline_state else 0,
            dataframe_json=df.to_json(orient="split", default_handler=str),
            column_schema=schema,
            error_log=error_log,
            previous_actions=list(self._get_history()),
            hint=hint,
            done=done,
            reward=float(reward),
            metadata=metadata or {},
        )

    def _get_history(self) -> list[str]:
        return self._history

    def _maybe_hint(self, reward: float) -> Optional[str]:
        step = self._pipeline_state.step_count if self._pipeline_state else 0
        if step >= HINT_UNLOCK_AFTER and reward < 0.30:
            return self._fixture.hint
        return None