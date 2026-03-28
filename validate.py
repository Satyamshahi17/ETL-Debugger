"""
run_tests.py  —  standalone spec compliance verifier
Runs all OpenEnv spec checks without requiring pytest.
Usage: python3 run_tests.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from io import StringIO

from models import PipelineAction, PipelineObservation, PipelineState
from server.pipeline_environment import PipelineEnvironment, MAX_STEPS
from server.reward import compute_reward
from tasks.fixtures import load_task
from graders.grader_easy   import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard   import grade as grade_hard

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[94m{}\033[0m"

results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"{FAIL}  {name}")
        print(f"       {msg}")
        results.append((name, False, msg))

# ─────────────────────────────────────────────
print(HEAD.format("\n── reset() ──────────────────────────────────"))

def t_reset_returns_obs():
    env = PipelineEnvironment("easy")
    obs = env.reset()
    assert isinstance(obs, PipelineObservation)

def t_reset_step_zero():
    env = PipelineEnvironment("easy")
    obs = env.reset()
    assert obs.step == 0

def t_reset_done_false():
    env = PipelineEnvironment("easy")
    obs = env.reset()
    assert obs.done is False

def t_reset_reward_zero():
    env = PipelineEnvironment("easy")
    obs = env.reset()
    assert obs.reward == 0.0

def t_reset_schema_populated():
    for tid in ["easy", "medium", "hard"]:
        env = PipelineEnvironment(tid)
        obs = env.reset()
        assert len(obs.schema) > 0
        for s in obs.schema:
            assert "name" in s and "dtype" in s

def t_reset_dataframe_valid():
    env = PipelineEnvironment("easy")
    obs = env.reset()
    df = pd.read_json(StringIO(obs.dataframe_json), orient="split")
    assert len(df) > 0

def t_reset_clears_state():
    env = PipelineEnvironment("easy")
    env.reset()
    env.step(PipelineAction(action_type="done"))
    obs2 = env.reset()
    assert obs2.step == 0 and obs2.done is False

for name, fn in [
    ("reset() returns PipelineObservation", t_reset_returns_obs),
    ("reset() step == 0", t_reset_step_zero),
    ("reset() done == False", t_reset_done_false),
    ("reset() reward == 0.0", t_reset_reward_zero),
    ("reset() schema populated for all 3 tasks", t_reset_schema_populated),
    ("reset() dataframe_json is valid JSON", t_reset_dataframe_valid),
    ("reset() clears previous episode state", t_reset_clears_state),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── step() ──────────────────────────────────"))

def t_step_returns_obs():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="done"))
    assert isinstance(obs, PipelineObservation)

def t_step_done_action_ends_episode():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="done"))
    assert obs.done is True

def t_step_count_increments():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="cast_type",
                                  column="revenue", params={"dtype":"str"}))
    assert obs.step == 1

def t_step_reward_in_range():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="done"))
    assert -0.30 <= obs.reward <= 1.0, f"reward {obs.reward} out of range"

def t_step_max_steps_ends():
    env = PipelineEnvironment("easy"); env.reset()
    obs = None
    for _ in range(MAX_STEPS + 5):
        obs = env.step(PipelineAction(action_type="cast_type",
                                      column="revenue", params={"dtype":"str"}))
        if obs.done: break
    assert obs.done is True

def t_step_after_done_raises():
    env = PipelineEnvironment("easy"); env.reset()
    env.step(PipelineAction(action_type="done"))
    try:
        env.step(PipelineAction(action_type="done"))
        assert False, "should have raised"
    except RuntimeError:
        pass

def t_step_before_reset_raises():
    env = PipelineEnvironment("easy")
    try:
        env.step(PipelineAction(action_type="done"))
        assert False, "should have raised"
    except RuntimeError:
        pass

def t_step_invalid_action_logs_error():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="rename_column",
                                  column="no_such_col", params={"new_name":"x"}))
    assert len(obs.error_log) > 0

def t_step_metadata_has_breakdown():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="done"))
    assert "schema_match" in obs.metadata
    assert "row_correctness" in obs.metadata
    assert "total" in obs.metadata

def t_step_previous_actions_tracked():
    env = PipelineEnvironment("easy"); env.reset()
    env.step(PipelineAction(action_type="cast_type", column="revenue",
                            params={"dtype":"str"}))
    obs = env.step(PipelineAction(action_type="done"))
    assert "cast_type" in obs.previous_actions
    assert "done" in obs.previous_actions

for name, fn in [
    ("step() returns PipelineObservation", t_step_returns_obs),
    ("step(done) sets obs.done=True", t_step_done_action_ends_episode),
    ("step() increments obs.step", t_step_count_increments),
    ("step() reward in [-0.30, 1.0]", t_step_reward_in_range),
    ("step() ends episode at MAX_STEPS", t_step_max_steps_ends),
    ("step() after done raises RuntimeError", t_step_after_done_raises),
    ("step() before reset raises RuntimeError", t_step_before_reset_raises),
    ("step() invalid action populates error_log", t_step_invalid_action_logs_error),
    ("step() metadata contains reward breakdown", t_step_metadata_has_breakdown),
    ("step() previous_actions tracks history", t_step_previous_actions_tracked),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── state property ────────────────────────────"))

def t_state_before_reset_raises():
    env = PipelineEnvironment("easy")
    try:
        _ = env.state
        assert False, "should have raised"
    except RuntimeError:
        pass

def t_state_returns_pipeline_state():
    env = PipelineEnvironment("easy"); env.reset()
    assert isinstance(env.state, PipelineState)

def t_state_has_episode_id():
    env = PipelineEnvironment("easy"); env.reset()
    assert env.state.episode_id != ""

def t_state_step_count_matches():
    env = PipelineEnvironment("easy"); env.reset()
    obs = env.step(PipelineAction(action_type="done"))
    assert env.state.step_count == obs.step

def t_state_task_id_correct():
    for tid in ["easy", "medium", "hard"]:
        env = PipelineEnvironment(tid); env.reset()
        assert env.state.task_id == tid

def t_state_is_done_after_done():
    env = PipelineEnvironment("easy"); env.reset()
    env.step(PipelineAction(action_type="done"))
    assert env.state.is_done is True

for name, fn in [
    ("state before reset raises RuntimeError", t_state_before_reset_raises),
    ("state returns PipelineState", t_state_returns_pipeline_state),
    ("state.episode_id is non-empty string", t_state_has_episode_id),
    ("state.step_count matches obs.step", t_state_step_count_matches),
    ("state.task_id correct for all 3 tasks", t_state_task_id_correct),
    ("state.is_done=True after done action", t_state_is_done_after_done),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── reward function ───────────────────────────"))

def t_reward_loop_penalty():
    fixture = load_task("easy")
    broken, gt = fixture.broken_df, fixture.ground_truth_df
    r_loop   = compute_reward(broken, gt, ["cast_type", "cast_type", "cast_type"])
    r_no_loop = compute_reward(broken, gt, ["cast_type"])
    # Loop penalty (-0.30) must make looped reward strictly lower
    assert r_loop < r_no_loop, f"loop ({r_loop}) should be < no-loop ({r_no_loop})"
    # And the penalty must have fired (difference >= 0.25)
    assert (r_no_loop - r_loop) >= 0.25, f"penalty too small: diff={r_no_loop - r_loop:.3f}"

def t_reward_perfect_df_scores_high():
    fixture = load_task("easy")
    gt = fixture.ground_truth_df
    r = compute_reward(gt, gt, ["fix_column", "cast_type"])
    assert r >= 0.85, f"expected >= 0.85 for perfect df, got {r}"

def t_reward_clipped():
    fixture = load_task("easy")
    r = compute_reward(fixture.broken_df, fixture.ground_truth_df, [])
    assert -0.30 <= r <= 1.0

def t_reward_partial_credit():
    fixture = load_task("easy")
    broken = fixture.broken_df.copy()
    gt = fixture.ground_truth_df
    r_before = compute_reward(broken, gt, [])
    # fix just the column names (partial improvement)
    broken.columns = gt.columns
    r_after = compute_reward(broken, gt, ["rename_column"])
    assert r_after >= r_before, "partial fix should not decrease reward"

for name, fn in [
    ("loop penalty fires when last 3 actions identical", t_reward_loop_penalty),
    ("perfect DataFrame scores >= 0.85", t_reward_perfect_df_scores_high),
    ("reward clipped to [-0.30, 1.0]", t_reward_clipped),
    ("partial fix >= no fix (monotone progress)", t_reward_partial_credit),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── graders ───────────────────────────────────"))

def t_grader_easy_correct_seq():
    env = PipelineEnvironment("easy"); env.reset()
    env.step(PipelineAction(action_type="fix_column", column="revenue",
                            params={"transform": "strip_currency"}))
    obs = env.step(PipelineAction(action_type="cast_type", column="revenue",
                                  params={"dtype": "float64"}))
    env.step(PipelineAction(action_type="done"))
    final_df = pd.read_json(StringIO(obs.dataframe_json), orient="split")
    score = grade_easy(final_df, env._fixture.ground_truth_df)
    assert score >= 0.80, f"easy correct sequence expected >= 0.80, got {score}"

def t_grader_easy_zero_on_empty():
    empty = pd.DataFrame()
    gt = load_task("easy").ground_truth_df
    assert grade_easy(empty, gt) == 0.0

def t_grader_easy_partial_credit():
    fixture = load_task("easy")
    broken = fixture.broken_df.copy()
    broken["revenue"] = broken["revenue"].str.replace(r"[\$,]", "", regex=True)
    score = grade_easy(broken, fixture.ground_truth_df)
    # String stripped but not cast yet — partial credit between 0 and full
    assert 0.0 < score < 1.0, f"expected partial credit, got {score}"

def t_grader_medium_returns_float():
    fixture = load_task("medium")
    score = grade_medium(fixture.broken_df, fixture.ground_truth_df)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def t_grader_hard_returns_float():
    fixture = load_task("hard")
    score = grade_hard(fixture.broken_df, fixture.ground_truth_df)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def t_graders_gt_scores_one():
    for tid, grader in [("easy", grade_easy), ("medium", grade_medium), ("hard", grade_hard)]:
        fixture = load_task(tid)
        score = grader(fixture.ground_truth_df, fixture.ground_truth_df)
        assert score >= 0.95, f"{tid}: ground truth vs itself should score >= 0.95, got {score}"

def t_grader_difficulty_ordering():
    """Broken DF should score lower on harder tasks than easier ones."""
    scores = {}
    for tid, grader in [("easy", grade_easy), ("medium", grade_medium), ("hard", grade_hard)]:
        fixture = load_task(tid)
        scores[tid] = grader(fixture.broken_df, fixture.ground_truth_df)
    # All broken DFs should score < 0.8 (none are trivially solved)
    for tid, s in scores.items():
        assert s < 0.8, f"{tid} broken df scored too high: {s}"

for name, fn in [
    ("easy grader: correct sequence scores >= 0.80", t_grader_easy_correct_seq),
    ("easy grader: empty df returns 0.0", t_grader_easy_zero_on_empty),
    ("easy grader: partial fix gets partial credit", t_grader_easy_partial_credit),
    ("medium grader: returns float in [0,1]", t_grader_medium_returns_float),
    ("hard grader: returns float in [0,1]", t_grader_hard_returns_float),
    ("all graders: ground_truth vs itself >= 0.95", t_graders_gt_scores_one),
    ("all graders: broken df scores < 0.80", t_grader_difficulty_ordering),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── action dispatcher ─────────────────────────"))

def t_action_fix_column_strip_currency():
    from server.actions import apply_action
    df = pd.DataFrame({"revenue": ["$1,234.56", "$0.99"]})
    out = apply_action(df, PipelineAction("fix_column", "revenue",
                                          {"transform": "strip_currency"}))
    assert out["revenue"].iloc[0] == "1234.56"

def t_action_cast_type_float():
    from server.actions import apply_action
    df = pd.DataFrame({"revenue": ["1234.56", "0.99"]})
    out = apply_action(df, PipelineAction("cast_type", "revenue",
                                          {"dtype": "float64"}))
    assert out["revenue"].dtype == "float64"

def t_action_rename_column():
    from server.actions import apply_action
    df = pd.DataFrame({"userId": [1, 2]})
    out = apply_action(df, PipelineAction("rename_column", "userId",
                                          {"new_name": "user_id"}))
    assert "user_id" in out.columns
    assert "userId" not in out.columns

def t_action_fill_nulls_zero():
    from server.actions import apply_action
    df = pd.DataFrame({"v": [1.0, None, 3.0]})
    out = apply_action(df, PipelineAction("fill_nulls", "v",
                                          {"strategy": "zero"}))
    assert out["v"].isnull().sum() == 0
    assert out["v"].iloc[1] == 0.0

def t_action_bad_column_raises():
    from server.actions import apply_action
    df = pd.DataFrame({"a": [1]})
    try:
        apply_action(df, PipelineAction("rename_column", "no_such",
                                        {"new_name": "x"}))
        assert False, "should raise"
    except KeyError:
        pass

def t_action_done_is_noop():
    from server.actions import apply_action
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = apply_action(df, PipelineAction("done"))
    assert list(out["a"]) == [1, 2, 3]

def t_action_invalid_type_raises():
    try:
        PipelineAction(action_type="teleport")
        assert False, "should raise ValueError"
    except ValueError:
        pass

for name, fn in [
    ("fix_column strip_currency removes $, commas", t_action_fix_column_strip_currency),
    ("cast_type converts str to float64", t_action_cast_type_float),
    ("rename_column renames correctly", t_action_rename_column),
    ("fill_nulls zero strategy eliminates NaNs", t_action_fill_nulls_zero),
    ("bad column name raises KeyError", t_action_bad_column_raises),
    ("done action is a no-op on DataFrame", t_action_done_is_noop),
    ("invalid action_type raises ValueError", t_action_invalid_type_raises),
]:
    check(name, fn)

# ─────────────────────────────────────────────
print(HEAD.format("\n── openenv.yaml ──────────────────────────────"))

def t_yaml_exists():
    assert os.path.exists("openenv.yaml")

def t_yaml_has_required_keys():
    import re
    with open("openenv.yaml") as f:
        content = f.read()
    for key in ["name:", "version:", "tasks:", "reward:", "observation_space:",
                "action_space:", "docker:", "baseline:"]:
        assert key in content, f"openenv.yaml missing key: {key}"

def t_yaml_three_tasks():
    import re
    with open("openenv.yaml") as f:
        content = f.read()
    task_ids = re.findall(r"  - id: (\w+)", content)
    assert set(task_ids) == {"easy", "medium", "hard"}

for name, fn in [
    ("openenv.yaml exists", t_yaml_exists),
    ("openenv.yaml has all required top-level keys", t_yaml_has_required_keys),
    ("openenv.yaml defines exactly 3 tasks", t_yaml_three_tasks),
]:
    check(name, fn)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"\n{'─'*52}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗ {name}")
            print(f"      {err}")
else:
    print("  — all tests passed ✓")
print(f"{'─'*52}\n")

sys.exit(0 if failed == 0 else 1)