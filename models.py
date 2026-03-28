"""
models.py — Pipeline Debug Environment
Typed Action, Observation, and State models following the OpenEnv spec.

Per RFC 002: Observation base class carries `done`, `reward`, and `metadata`.
Models use Python dataclasses extending openenv.core.env_server base types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    # Fallback base classes when openenv is not installed (for local dev / testing)
    from dataclasses import dataclass as _dc

    @_dc(kw_only=True)
    class Observation:
        done: bool = False
        reward: Union[bool, int, float, None] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @_dc
    class Action:
        pass

    @_dc
    class State:
        episode_id: str = ""
        step_count: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "fix_column",
    "cast_type",
    "drop_rows",
    "rename_column",
    "fill_nulls",
    "split_column",
    "merge_columns",
    "reorder_columns",
    "done",
}


@dataclass
class PipelineAction(Action):
    """
    A single corrective operation the agent applies to the broken DataFrame.

    action_type : one of VALID_ACTION_TYPES
    column      : target column name (required for most actions)
    params      : action-specific keyword arguments, e.g.
                    cast_type    → {"dtype": "float64"}
                    fill_nulls   → {"strategy": "mean" | "zero" | "forward"}
                    rename_column→ {"new_name": "revenue"}
                    split_column → {"delimiter": "_", "new_cols": ["a","b"]}
                    drop_rows    → {"condition": "revenue < 0"}
                    fix_column   → {"transform": "strip_currency"}
    """

    action_type: str = "done"
    column: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type '{self.action_type}'. "
                f"Valid types: {sorted(VALID_ACTION_TYPES)}"
            )


# ---------------------------------------------------------------------------
# Observation  (reward + done live here per OpenEnv spec)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class PipelineObservation(Observation):
    """
    Full observation returned after reset() or step().

    task_id          : which task is running ("easy" | "medium" | "hard")
    step             : current step number (0-indexed at reset)
    dataframe_json   : current DataFrame as JSON string (orient="split")
    schema           : list of {name, dtype, nullable, n_nulls, sample} dicts
    error_log        : exceptions caught during last action (empty = clean)
    previous_actions : history of action_type strings this episode
    hint             : optional hint string, unlocked after 5 failed steps

    Inherited from Observation base:
        done     : bool  — True when episode has ended
        reward   : float — cumulative reward for this step (0.0 – 1.0)
        metadata : dict  — extra diagnostic info
    """

    task_id: str = ""
    step: int = 0
    dataframe_json: str = ""
    column_schema: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)
    previous_actions: List[str] = field(default_factory=list)
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# State  (episode metadata, returned by state())
# ---------------------------------------------------------------------------

@dataclass
class PipelineState(State):
    """
    Current episode state, returned by env.state property.

    Extends the base State (episode_id, step_count) with pipeline-specific
    tracking fields used internally by the environment.

    accumulated_reward : running reward total
    last_action_type   : most recent action taken
    consecutive_loops  : count of repeated identical action sequences
    task_id            : active task
    """

    accumulated_reward: float = 0.0
    last_action_type: str = ""
    consecutive_loops: int = 0
    task_id: str = ""
    is_done: bool = False