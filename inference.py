"""Baseline inference agent for PipelineRx.

Uses the OpenAI client to call an LLM that diagnoses and repairs broken ML
pipeline data through the PipelineRx REST API.

Required environment variables:
    API_BASE_URL  — LLM API endpoint (OpenAI-compatible)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — API key
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import requests
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data engineer debugging a broken ML data pipeline.
You receive observations about a corrupted pandas DataFrame and must diagnose and fix the issues.

Call exactly ONE action per turn. Respond with ONLY valid JSON — no explanation, no markdown.

Available actions:

1. inspect_data — View first N rows
   {"action": "inspect_data", "parameters": {"n_rows": 5}}

2. inspect_schema — View column names, dtypes, null counts
   {"action": "inspect_schema", "parameters": {}}

3. check_column_stats — View statistics for a specific column
   {"action": "check_column_stats", "parameters": {"column": "col_name"}}

4. apply_fix — Apply a targeted fix. Operations:
   - strip_and_cast: {"action": "apply_fix", "parameters": {"column": "col", "operation": "strip_and_cast", "strip_chars": "%$,", "target_dtype": "float64"}}
   - replace_value: {"action": "apply_fix", "parameters": {"column": "col", "operation": "replace_value", "old_value": "missing", "new_value": "NaN"}}
   - normalize_zscore: {"action": "apply_fix", "parameters": {"column": "col", "operation": "normalize_zscore"}}
   - normalize_minmax: {"action": "apply_fix", "parameters": {"column": "col", "operation": "normalize_minmax"}}
   - rename_column: {"action": "apply_fix", "parameters": {"operation": "rename_column", "old_name": "old", "new_name": "new"}}
   - drop_column: {"action": "apply_fix", "parameters": {"column": "col", "operation": "drop_column"}}
   - encode_onehot: {"action": "apply_fix", "parameters": {"column": "col", "operation": "encode_onehot"}}
   - cast_datetime_to_numeric: {"action": "apply_fix", "parameters": {"column": "col", "operation": "cast_datetime_to_numeric"}}
   - filter_values: {"action": "apply_fix", "parameters": {"column": "col", "operation": "filter_values", "value": "bad_val"}}
   - convert_timezone: {"action": "apply_fix", "parameters": {"column": "col", "operation": "convert_timezone", "offset_hours": 5.5}}
   - fix_stage (Task 5 only): {"action": "apply_fix", "parameters": {"operation": "fix_stage", "stage_id": 2, "fix_type": "fix_join"}}

5. run_tests — Run the grader and see your score
   {"action": "run_tests", "parameters": {}}

6. inspect_stage (Task 5 only) — Inspect a pipeline stage
   {"action": "inspect_stage", "parameters": {"stage_id": 1}}

Strategy:
1. First inspect_schema to understand the data
2. Then check_column_stats on suspicious columns
3. Apply fixes one at a time
4. Call run_tests after each fix to track progress
5. When reward >= 0.95 or all tests pass, stop

CRITICAL: Respond with ONLY a JSON object. No text before or after."""


# ── Sliding window for conversation history ─────────────────────────────────

MAX_HISTORY_MESSAGES = 16  # Keep last 8 exchanges (16 messages)


def trim_history(history: list) -> list:
    """Keep conversation history manageable to avoid context overflow."""
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]
    return history


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(task_id: int) -> float:
    """Run one episode for a given task. Returns final reward."""
    final_reward = 0.0
    done = False
    step = 0

    try:
        # Reset the environment
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        ).json()

        observation = reset_resp.get("observation", "Episode started.")
        # Read max_steps from the reset info so Task 5 gets 40 steps
        max_steps = reset_resp.get("info", {}).get("max_steps", 30)
        print(
            f'[START] task_id={task_id} observation="{observation[:80]}"',
            flush=True,
        )

        history = [{"role": "user", "content": observation}]

        for step in range(max_steps):
            # Trim history to prevent context overflow
            history = trim_history(history)

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                    max_tokens=600,
                    temperature=0.1,
                )
                agent_action_str = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[STEP] task_id={task_id} step={step+1} action=error reward={final_reward:.4f} done=false", flush=True)
                continue

            # Parse the LLM output as JSON
            try:
                # Handle markdown code blocks
                cleaned = agent_action_str
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    cleaned = "\n".join(lines)
                agent_action = json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from the response
                try:
                    start = agent_action_str.index("{")
                    end = agent_action_str.rindex("}") + 1
                    agent_action = json.loads(agent_action_str[start:end])
                except (ValueError, json.JSONDecodeError):
                    agent_action = {"action": "inspect_schema", "parameters": {}}

            # Ensure required keys
            if "action" not in agent_action:
                agent_action = {"action": "inspect_schema", "parameters": {}}
            if "parameters" not in agent_action:
                agent_action["parameters"] = {}

            # Execute the action
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json=agent_action,
                    timeout=30,
                ).json()
            except Exception as e:
                print(f"[STEP] task_id={task_id} step={step+1} action={agent_action.get('action', 'unknown')} reward={final_reward:.4f} done=false", flush=True)
                continue

            reward = step_resp.get("reward", 0.0)
            done = step_resp.get("done", False)
            obs = step_resp.get("observation", "")

            action_name = agent_action.get("action", "unknown")
            print(
                f"[STEP] task_id={task_id} step={step+1} action={action_name} "
                f"reward={reward:.4f} done={str(done).lower()}",
                flush=True,
            )

            # Update history
            history.append({"role": "assistant", "content": json.dumps(agent_action)})
            history.append({"role": "user", "content": obs})

            final_reward = reward

            if done:
                break

        # Force a final run_tests to ensure final_reward is the true grader score
        if not done:
            try:
                final_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": "run_tests", "parameters": {}},
                    timeout=30,
                ).json()
                final_reward = final_resp.get("reward", final_reward)
            except Exception:
                pass  # keep last known reward

    except Exception as e:
        traceback.print_exc(file=sys.stderr)

    # Always print [END] line — even on failure
    status = "success" if final_reward >= 0.95 else "partial" if final_reward >= 0.5 else "fail"
    print(
        f"[END] task_id={task_id} final_reward={final_reward:.4f} "
        f"steps={step + 1} status={status}",
        flush=True,
    )

    return final_reward


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify required env vars
    missing = []
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"WARNING: Missing environment variables: {missing}", file=sys.stderr)

    scores = []
    for t in range(1, 6):
        reward = run_episode(t)
        scores.append(reward)

    print(f"\nBaseline scores: {scores}", flush=True)
    print(f"Mean reward: {sum(scores)/len(scores):.4f}", flush=True)
