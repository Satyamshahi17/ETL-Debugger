"""
inference.py — ETLDebugger baseline inference script

Reads:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key

Usage:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o \
    HF_TOKEN=sk-... \
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from typing import List, Optional

import pandas as pd
from openai import OpenAI

# from groq import Groq
# from groq import APIStatusError
# from dotenv import load_dotenv

from models import PipelineAction
from server.pipeline_environment import PipelineEnvironment
from graders.grader_easy   import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard   import grade as grade_hard

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
# load_dotenv()
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TASKS   = ["easy", "medium", "hard"]
GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

SYSTEM_PROMPT = """\
You are a data engineering agent. You will receive a broken pandas DataFrame
and must fix it by issuing structured JSON actions one at a time.

Each action must be a JSON object with:
  - "action_type": one of fix_column | cast_type | drop_rows | rename_column |
                   fill_nulls | split_column | merge_columns | reorder_columns | done
  - "column": string column name (null if not applicable)
  - "params": dict of action-specific parameters

Common params:
  fix_column:      {"transform": "strip_currency"}
  cast_type:       {"dtype": "float64"}
  rename_column:   {"new_name": "user_id"}
  fill_nulls:      {"strategy": "zero"}
  drop_rows:       {"condition": "revenue < 0"}
  reorder_columns: {"order": ["col1", "col2"]}

When the DataFrame is fixed, issue {"action_type": "done"}.
Respond ONLY with a valid JSON object — no markdown, no explanation.
"""

# ---------------------------------------------------------------------------
# Mandatory Logging Functions
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error).replace('\n', ' ') if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# LLM call — isolated so failures never crash the episode
# ---------------------------------------------------------------------------

def get_action(messages: list):
    """Call the LLM and parse a PipelineAction. Returns action_type='done' on any failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        d   = json.loads(raw)
        return raw, PipelineAction(
            action_type=d.get("action_type", "done"),
            column=d.get("column"),
            params=d.get("params", {}),
        )
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return '{"action_type": "done"}', PipelineAction(action_type="done")

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> dict:
    env    = PipelineEnvironment(task_id)
    grader = GRADERS[task_id]
    obs    = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _obs_to_prompt(obs)},
    ]

    done            = False
    rewards_history: List[float] = []
    score           = 0.0
    is_success      = False
    steps_taken     = getattr(obs, "step", 0)

    log_start(task=task_id, env="ETLDebugger", model=MODEL_NAME)

    try:
        while not done:
            raw, action = get_action(messages)

            obs         = env.step(action)
            done        = obs.done
            steps_taken = obs.step

            current_reward = obs.reward or 0.0
            rewards_history.append(current_reward)

            error_msg = obs.error_log[-1] if hasattr(obs, 'error_log') and obs.error_log else None

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": _obs_to_prompt(obs)})

            log_step(step=obs.step, action=action.action_type, reward=current_reward, done=done, error=error_msg)

        final_df   = pd.read_json(StringIO(obs.dataframe_json), orient="split")
        score      = grader(final_df, env._fixture.ground_truth_df)
        score      = min(max(score, 0.0), 1.0)
        is_success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task({task_id}) exception: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=is_success, steps=steps_taken, score=score, rewards=rewards_history)

    return {"task_id": task_id, "score": score, "steps": steps_taken}


def _obs_to_prompt(obs) -> str:
    schema_text = "\n".join(
        f"  {s['name']:20s} dtype={s['dtype']:10s} nulls={s['n_nulls']}"
        for s in obs.column_schema
    )
    errors = "\n".join(obs.error_log) if obs.error_log else "None"
    hint   = f"\nHint: {obs.hint}" if obs.hint else ""
    return (
        f"Task: {obs.task_id} | Step: {obs.step} | Reward: {obs.reward:.3f}\n"
        f"Previous actions: {obs.previous_actions}\n\n"
        f"Schema:\n{schema_text}\n\nErrors: {errors}{hint}"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # print(f"Tasks : {TASKS}", file=sys.stderr)

    results = {}
    for task_id in TASKS:
        # print(f"── {task_id.upper()} ──", file=sys.stderr)
        try:
            result = run_task(task_id)
        except Exception as exc:
            # Belt-and-suspenders: run_task catches internally, but if something
            # escapes (e.g. env.reset() itself crashes), still continue.
            print(f"[DEBUG] Unhandled in run_task({task_id}): {exc}", file=sys.stderr, flush=True)
            result = {"task_id": task_id, "score": 0.0, "steps": 0}
        results[task_id] = result
        # print(f"   score={result['score']:.4f}  steps={result['steps']}", file=sys.stderr)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # print("── SUMMARY ──", file=sys.stderr)
    # for task_id, r in results.items():
    #     print(f"  {task_id:8s}  {r['score']:.4f}", file=sys.stderr)
    # print("Results saved to results.json", file=sys.stderr)

    # for task_id, r in results.items():
    #     assert 0.0 <= r["score"] <= 1.0, f"{task_id} score out of range: {r['score']}"
    # print("All scores in valid range [0.0, 1.0] ✓", file=sys.stderr)