"""
inference.py — ETLDebugger baseline inference script
Required by hackathon pre-submission checklist.

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

import pandas as pd
from openai import OpenAI

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
    print("ERROR: HF_TOKEN environment variable not set.")
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

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

    done = False
    while not done:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content

        try:
            d      = json.loads(raw)
            action = PipelineAction(
                action_type=d.get("action_type", "done"),
                column=d.get("column"),
                params=d.get("params", {}),
            )
        except Exception:
            action = PipelineAction(action_type="done")

        obs  = env.step(action)
        done = obs.done

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": _obs_to_prompt(obs)})

        print(f"  [{task_id}] step={obs.step} action={action.action_type} reward={obs.reward:.3f}")

    final_df = pd.read_json(StringIO(obs.dataframe_json), orient="split")
    score    = grader(final_df, env._fixture.ground_truth_df)
    return {"task_id": task_id, "score": score, "steps": obs.step}


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
    print(f"\nModel : {MODEL_NAME}")
    print(f"Base  : {API_BASE_URL}")
    print(f"Tasks : {TASKS}\n")

    results = {}
    for task_id in TASKS:
        print(f"── {task_id.upper()} ──")
        result = run_task(task_id)
        results[task_id] = result
        print(f"   score={result['score']:.4f}  steps={result['steps']}\n")

    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("── SUMMARY ──")
    for task_id, r in results.items():
        print(f"  {task_id:8s}  {r['score']:.4f}")
    print("\nResults saved to results.json")

    # Fail loudly if any score is out of range
    for task_id, r in results.items():
        assert 0.0 <= r["score"] <= 1.0, f"{task_id} score out of range: {r['score']}"
    print("All scores in valid range [0.0, 1.0] ✓")