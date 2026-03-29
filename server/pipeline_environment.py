"""
server/pipeline_environment.py
Core OpenEnv-compliant Environment for the Data Pipeline Debugging task.

Implements the three mandatory methods:
    reset()  → PipelineObservation   (initial observation)
    step()   → PipelineObservation   (observation with reward + done)
    state    → PipelineState         (@property, episode metadata)

Per RFC 002:
  - reward lives on the Observation (obs.reward)
  - done lives on the Observation   (obs.done)
  - state() provides episode metadata (step_count, episode_id, etc.)
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

import pandas as pd

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

    class Environment:
        """Minimal stand-in when openenv package is not installed."""
        pass

    from dataclasses import dataclass as _state_dc
    @_state_dc
    class State:
        episode_id: str = ""
        step_count: int = 0

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import PipelineAction, PipelineObservation, PipelineState
from tasks.fixtures import load_task
from server.actions import apply_action
from server.reward import compute_reward, compute_reward_breakdown

MAX_STEPS = 20
HINT_UNLOCK_AFTER = 5   # steps with reward < 0.3 before hint is shown


class PipelineEnvironment(Environment):
    """
    Data Pipeline Debugging Environment.

    The agent receives a broken DataFrame and must apply a sequence of
    corrective PipelineActions to make it match the ground truth.

    Usage:
        env = PipelineEnvironment("easy")
        obs = env.reset()
        obs = env.step(PipelineAction(action_type="fix_column", ...))
        ...
        print(env.state.accumulated_reward)
    """

    def __init__(self, task_id: str = "easy"):
        if _OPENENV_AVAILABLE:
            super().__init__()
        self.task_id = task_id
        self._fixture = load_task(task_id)
        self._df: Optional[pd.DataFrame] = None
        self._state: Optional[PipelineState] = None

    # ------------------------------------------------------------------
    # OpenEnv mandatory interface
    # ------------------------------------------------------------------

    def reset(self) -> PipelineObservation:
        """
        Initialise a new episode.
        Returns the initial observation of the broken DataFrame.
        """
        self._df = self._fixture.broken_df.copy()
        self._state = PipelineState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            accumulated_reward=0.0,
            last_action_type="",
            consecutive_loops=0,
            task_id=self.task_id,
            is_done=False,
        )
        return self._build_obs(
            error_log=[],
            reward=0.0,
            done=False,
        )

    def step(self, action: PipelineAction) -> PipelineObservation:
        """
        Execute one action and return the resulting observation.
        Reward and done are embedded in the returned Observation per spec.
        """
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_count += 1
        errors: list[str] = []

        # Apply action (catch errors, don't crash the episode)
        if action.action_type != "done":
            try:
                self._df = apply_action(self._df, action)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        # Track action history on state
        self._state.last_action_type = action.action_type
        history = self._get_history()
        history.append(action.action_type)
        self._state.__dict__["_history"] = history

        # Compute dense reward
        reward = compute_reward(
            self._df,
            self._fixture.ground_truth_df,
            history,
        )
        breakdown = compute_reward_breakdown(
            self._df,
            self._fixture.ground_truth_df,
            history,
        )
        self._state.accumulated_reward += reward

        # Episode end conditions
        done = (
            action.action_type == "done"
            or self._state.step_count >= MAX_STEPS
        )
        self._state.is_done = done

        # Unlock hint after prolonged struggle
        hint = self._maybe_hint(reward)

        return self._build_obs(
            error_log=errors,
            reward=reward,
            done=done,
            hint=hint,
            metadata=breakdown,
        )

    @property
    def state(self) -> PipelineState:
        """
        Returns current episode metadata.
        Per RFC 002, state() is a property on the server-side Environment.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        error_log: list,
        reward: float,
        done: bool,
        hint: str | None = None,
        metadata: dict | None = None,
    ) -> PipelineObservation:
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
            step=self._state.step_count if self._state else 0,
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
        return getattr(self._state, "_history", [])

    def _maybe_hint(self, reward: float) -> str | None:
        """Unlock the hint if the agent has been struggling."""
        step = self._state.step_count if self._state else 0
        if step >= HINT_UNLOCK_AFTER and reward < 0.30:
            return self._fixture.hint
        return None