"""
models.py — ETLDebugger
OpenEnv-compliant models using Pydantic per official docs.
Imports from openenv.core.env_server.types as specified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)


VALID_ACTION_TYPES = {
    "fix_column", "cast_type", "drop_rows", "rename_column",
    "fill_nulls", "split_column", "merge_columns", "reorder_columns", "done",
}


class PipelineAction(Action):
    """Corrective operation applied to the broken DataFrame."""
    action_type: str = "done"
    column: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type '{self.action_type}'. "
                f"Valid: {sorted(VALID_ACTION_TYPES)}"
            )


class PipelineObservation(Observation):
    """Full observation returned after reset() or step()."""
    task_id: str = ""
    step: int = 0
    dataframe_json: str = ""
    column_schema: List[Dict[str, Any]] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    previous_actions: List[str] = Field(default_factory=list)
    hint: Optional[str] = None


# State is used directly from openenv per official docs — not subclassed
# PipelineState is a plain Pydantic model for extra fields
from pydantic import BaseModel

class PipelineState(BaseModel):
    """Episode metadata returned by env.state property."""
    episode_id: str = ""
    step_count: int = 0
    accumulated_reward: float = 0.0
    last_action_type: str = ""
    task_id: str = ""
    is_done: bool = False