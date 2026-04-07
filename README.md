---
title: ETLDebugger
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/Dockerfile
pinned: false
---

# ETLDebugger

> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment where agents learn to debug broken data pipelines.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4CAF50?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![HF Spaces](https://img.shields.io/badge/HF%20Spaces-deployed-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

---

## Problem Statement

Data engineers spend a significant portion of their time debugging ETL pipelines — wrong column types loaded from CSVs, silent join failures due to key mismatches, and aggregation bugs that produce plausible-looking but incorrect results. These are real, high-stakes errors that cost companies millions in downstream decisions made on bad data.

ETLDebugger provides a structured, episodic environment where an agent must inspect a broken DataFrame, reason about what went wrong, and apply a sequence of corrective actions to restore it to ground truth.

---

## The Solution

ETLDebugger wraps three carefully designed data corruption scenarios into a fully OpenEnv-compliant environment. The agent receives a broken `pandas` DataFrame as an observation, chooses from a structured action space (cast types, rename columns, fix values, drop rows, etc.), and receives a dense reward signal after each step based on how close the current DataFrame is to the ground truth.

No external APIs, no databases, no network calls — all fixtures are generated deterministically in code, making the environment fully reproducible and lightweight enough to run on a laptop.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RL Agent / Inference                  │
│              (OpenAI client via inference.py)            │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP (reset / step / state)
                        ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Server  (server/app.py)            │
│                                                         │
│   POST /reset  ──►  PipelineEnvironment.reset()         │
│   POST /step   ──►  PipelineEnvironment.step()          │
│   GET  /state  ──►  PipelineEnvironment.state           │
│   GET  /health ──►  { status: ok }                      │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Task Fixture    Action         Reward
    (fixtures.py)  Dispatcher    Function
                   (actions.py)  (reward.py)
          │
          ├── easy_broken_df   ──► grader_easy.py
          ├── medium_broken_df ──► grader_medium.py
          └── hard_broken_df   ──► grader_hard.py
```

---

## Tasks

ETLDebugger ships with three tasks of increasing difficulty, each with a deterministic grader scoring 0.0 – 1.0.

### Task 1 — Type Mismatch `easy`

**What broke:** The `revenue` column was loaded from CSV as a currency-formatted string (`"$1,234.56"`) instead of `float64`. The pipeline never cast it.

**What the agent must do:** Strip the `$` and commas, then cast to float.

**Optimal sequence:**
```json
{"action_type": "fix_column",  "column": "revenue", "params": {"transform": "strip_currency"}}
{"action_type": "cast_type",   "column": "revenue", "params": {"dtype": "float64"}}
{"action_type": "done"}
```

**Llama-3.1-8b-instant score:** `1.00`

---

### Task 2 — Join Key Mismatch `medium`

**What broke:** A left-join between `orders` and `users` failed because the key column is `user_id` in orders but `userId` (camelCase) in users. The result has a `userId` artifact column and `name`/`tier` columns that are entirely `NULL`.

**What the agent must do:** Identify the camelCase mismatch, rename `userId` → `user_id`, and reorder columns to match the expected schema.

**Optimal sequence:**
```json
{"action_type": "rename_column",   "column": "userId", "params": {"new_name": "user_id"}}
{"action_type": "reorder_columns", "params": {"order": ["user_id","name","tier","amount","status"]}}
{"action_type": "done"}
```

**Llama-3.1-8b-instant score:** `0.76`

---

### Task 3 — Silent Data Corruption `hard`

**What broke:** A date-range filter used `>` instead of `>=`, silently dropping the boundary row. Additionally, all `daily_total` values for category `D` were zeroed out in the aggregation step. No exception is raised — the output looks plausible.

**What the agent must do:** Audit row counts, detect the statistical anomaly in category D, and identify both bugs without any error message to guide it.

**Llama-3.1-8b-instant score:** `0.69`

---

## Reward Function

Every `step()` returns a dense reward in `[-0.30, 1.0]`. The reward is embedded directly on the `Observation` object per the OpenEnv spec.

```
reward = 0.40 × schema_match       # column names + dtypes vs ground truth
       + 0.30 × row_correctness    # % of cell values matching ground truth
       + 0.20 × null_handling      # null distribution vs ground truth
       + 0.10 × efficiency         # penalises redundant repeated actions
       − 0.30  (loop penalty)      # flat penalty if last 3 actions are identical
```

**Why dense rewards matter:** Binary end-of-episode rewards give the agent no signal until it solves the task completely. ETLDebugger rewards every meaningful step — fixing the dtype gives +0.12 immediately, even if nulls are still wrong. This makes learning tractable for RL algorithms.

**Loop penalty:** If the agent repeats the same action three times in a row, it receives a flat −0.30 penalty. This prevents the degenerate policy of spamming one action.

---

## Graders

Each task has a standalone deterministic grader in `graders/`. Graders are called at the end of an episode and return a `float` in `[0.0, 1.0]`.

| Grader | Checks | Partial credit |
|--------|--------|----------------|
| `grader_easy.py` | column exists, dtype is float64, value match to 2dp, no nulls introduced | yes — per check |
| `grader_medium.py` | user_id present (not userId), name/tier not null, row count, value correctness | yes — per check |
| `grader_hard.py` | row count within 2%, date range correctness, category D totals fixed, amount distribution | yes — per check |

All graders are **deterministic and reproducible**.

---

## Fixtures

All task data is generated programmatically in `tasks/fixtures.py` using a seeded random number generator (`numpy.random.default_rng(seed)`). This means:

- No external CSV files to maintain
- Fully reproducible across machines
- Each task's broken DataFrame is a precise, controlled corruption of the ground truth

| Task | Rows | Corruption type | Seed |
|------|------|-----------------|------|
| easy | 50 | Revenue column formatted as `"$X,XXX.XX"` string | 42 |
| medium | 80 | Join key `userId` vs `user_id` → all-NULL name/tier | 7 |
| hard | 200 | Date filter off-by-one + category D totals zeroed | 99 |

---

## Action Space

| `action_type` | Key params | Description |
|---|---|---|
| `fix_column` | `transform` | Strip currency symbols, whitespace, change case |
| `cast_type` | `dtype` | Cast to `float64`, `int64`, `str`, `datetime` |
| `drop_rows` | `condition` or `drop_nulls` | Remove rows by pandas query or null check |
| `rename_column` | `new_name` | Rename a column |
| `fill_nulls` | `strategy` or `value` | Fill NaNs with mean/median/zero/ffill |
| `split_column` | `delimiter`, `new_cols` | Split string column into multiple |
| `merge_columns` | `other_cols`, `separator` | Concatenate columns |
| `reorder_columns` | `order` | Reorder column list |
| `done` | — | Signal episode complete |

---

## Observation Space

Each `step()` and `reset()` returns a `PipelineObservation`:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task (`easy` / `medium` / `hard`) |
| `step` | `int` | Current step number |
| `dataframe_json` | `str` | Current DataFrame as JSON (`orient="split"`) |
| `column_schema` | `list[dict]` | Column names, dtypes, null counts, sample values |
| `error_log` | `list[str]` | Exceptions from last action |
| `previous_actions` | `list[str]` | Action history this episode |
| `hint` | `str \| None` | Unlocked after 5 steps with reward < 0.30 |
| `done` | `bool` | True when episode has ended |
| `reward` | `float` | Step reward (−0.30 to 1.0) |
| `metadata` | `dict` | Reward sub-score breakdown |

---

## Tech Stack

| Component | Technology |
|---|---|
| Environment core | Python 3.10+, pandas 2.x, numpy |
| API server | FastAPI + Uvicorn |
| Data models | OpenEnv dataclasses (Action, Observation, State) |
| Containerisation | Docker (`openenv-base:latest`) |
| Deployment | Hugging Face Spaces (Docker SDK) |
| Inference | OpenAI Python client (any OpenAI-compatible endpoint) |
| Package manager | uv |

---

## Repo Structure

```
etl-debugger/
├── inference.py              # Baseline inference script (required)
├── validate.py               # Spec compliance test runner (44 checks)
├── openenv.yaml              # OpenEnv metadata
├── pyproject.toml            # Package config + entry points
├── requirements.txt          # Dependencies
├── README.md
├── setup.sh                  # One-command setup script
├── __init__.py               # Exports PipelineAction, PipelineObservation
├── models.py                 # Typed Action / Observation / State dataclasses
├── client.py                 # HTTPEnvClient subclass
├── server/
│   ├── app.py                # FastAPI server (reset / step / state / health)
│   ├── pipeline_environment.py  # Core environment logic
│   ├── actions.py            # Action dispatcher
│   ├── reward.py             # Dense reward function
│   ├── Dockerfile            # openenv-base:latest
│   └── requirements.txt      # Server-specific deps
├── tasks/
│   └── fixtures.py           # Generates all 3 task fixtures deterministically
└── graders/
    ├── grader_easy.py
    ├── grader_medium.py
    └── grader_hard.py
```

---

## Setup & Usage

### Local

```bash
git clone https://github.com/YOUR_USERNAME/etl-debugger
cd etl-debugger
```

Or manually:

```bash
pip install -r requirements.txt
uv sync
uvicorn server.app:app --reload
```

### Validate spec compliance

```bash
python validate.py
# Results: 44/44 passed — all tests passed ✓
```

### API endpoints

```bash
# Reset episode
curl -X POST http://localhost:8000/reset

# Take an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"fix_column","column":"revenue","params":{"transform":"strip_currency"}}'

# Get episode state
curl http://localhost:8000/state

# Interactive docs
open http://localhost:8000/docs
```

### Run inference

```bash
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o \
HF_TOKEN=sk-... \
python inference.py
```

### Docker

```bash
docker build -f server/Dockerfile -t etl-debugger .
docker run -p 8000:8000 -e TASK_ID=easy etl-debugger
```
## OpenEnv Compliance

| Requirement | Status |
|---|---|
| Typed `Action`, `Observation`, `State` models | ✅ |
| `reset()` → initial observation | ✅ |
| `step(action)` → observation with reward + done | ✅ |
| `state` property → episode metadata | ✅ |
| `openenv.yaml` with metadata | ✅ |
| 3+ tasks with deterministic graders | ✅ |
| Dense reward function (not binary) | ✅ |
| Dockerfile builds cleanly | ✅ |
| HF Space deployment | ✅ |
| Baseline inference script (`inference.py`) | ✅ |

---

## Author

**Satyam Kumar**

---

## License

MIT
