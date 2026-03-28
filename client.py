"""
client.py — PipelineDebugEnv client
Implements HTTPEnvClient[PipelineAction, PipelineObservation] per the OpenEnv spec.

Usage:
    # Connect to a running server
    client = PipelineDebugEnv(base_url="http://localhost:8000")
    with client:
        result = client.reset()
        result = client.step(PipelineAction(action_type="fix_column",
                                            column="revenue",
                                            params={"transform": "strip_currency"}))
        print(result.observation.reward)

    # Or pull from Docker image
    client = PipelineDebugEnv.from_docker_image("etl-debugger:latest")

    # Or pull from HF Space
    client = PipelineDebugEnv.from_hub("your-org/etl-debugger")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models import PipelineAction, PipelineObservation, PipelineState

try:
    from openenv.core.http_env_client import HTTPEnvClient  # type: ignore
    from openenv.core.types import StepResult  # type: ignore

    class PipelineDebugEnv(HTTPEnvClient[PipelineAction, PipelineObservation]):
        """
        HTTP client for the ETLDebugger.
        Connects to a running FastAPI server (local or HF Space).
        """

        def _step_payload(self, action: PipelineAction) -> Dict[str, Any]:
            return {
                "action_type": action.action_type,
                "column":      action.column,
                "params":      action.params,
            }

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PipelineObservation]:
            obs = PipelineObservation(
                task_id=payload.get("task_id", ""),
                step=payload.get("step", 0),
                dataframe_json=payload.get("dataframe_json", ""),
                schema=payload.get("schema", []),
                error_log=payload.get("error_log", []),
                previous_actions=payload.get("previous_actions", []),
                hint=payload.get("hint"),
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                metadata=payload.get("metadata", {}),
            )
            return StepResult(
                observation=obs,
                reward=obs.reward,
                done=obs.done,
            )

        def _parse_state(self, payload: Dict[str, Any]) -> PipelineState:
            s = PipelineState(
                episode_id=payload.get("episode_id", ""),
                step_count=payload.get("step_count", 0),
                accumulated_reward=payload.get("accumulated_reward", 0.0),
                last_action_type=payload.get("last_action_type", ""),
                task_id=payload.get("task_id", ""),
                is_done=payload.get("is_done", False),
            )
            return s

except ImportError:
    # Fallback sync client using requests when openenv is not installed
    import requests

    class StepResult:
        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward      = reward
            self.done        = done

    class PipelineDebugEnv:
        """Minimal HTTP client for local dev without the openenv package."""

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

        def reset(self) -> StepResult:
            r = requests.post(f"{self.base_url}/reset")
            r.raise_for_status()
            return self._parse(r.json())

        def step(self, action: PipelineAction) -> StepResult:
            payload = {
                "action_type": action.action_type,
                "column":      action.column,
                "params":      action.params,
            }
            r = requests.post(f"{self.base_url}/step", json=payload)
            r.raise_for_status()
            return self._parse(r.json())

        def state(self) -> PipelineState:
            r = requests.get(f"{self.base_url}/state")
            r.raise_for_status()
            p = r.json()
            return PipelineState(
                episode_id=p.get("episode_id", ""),
                step_count=p.get("step_count", 0),
                accumulated_reward=p.get("accumulated_reward", 0.0),
                last_action_type=p.get("last_action_type", ""),
                task_id=p.get("task_id", ""),
                is_done=p.get("is_done", False),
            )

        def _parse(self, p: dict) -> StepResult:
            obs = PipelineObservation(
                task_id=p.get("task_id", ""),
                step=p.get("step", 0),
                dataframe_json=p.get("dataframe_json", ""),
                schema=p.get("schema", []),
                error_log=p.get("error_log", []),
                previous_actions=p.get("previous_actions", []),
                hint=p.get("hint"),
                done=p.get("done", False),
                reward=p.get("reward", 0.0),
                metadata=p.get("metadata", {}),
            )
            return StepResult(observation=obs, reward=obs.reward, done=obs.done)