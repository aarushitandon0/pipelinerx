---
title: PipelineRx
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
short_description: RL environment for ML pipeline debugging and repair
tags:
  - reinforcement-learning
  - openenv
  - data-engineering
  - machine-learning
  - environment
  - agent
  - fastapi
---

# PipelineRx

**An OpenEnv reinforcement learning environment for ML pipeline debugging.**

PipelineRx is a real-world RL environment where AI agents diagnose and repair silent ML pipeline failures. The agent interacts with a corrupted pandas DataFrame through a REST API, applying targeted fixes and receiving reward signals based on data quality. Five escalating incident scenarios test the agent across a full range of difficulty.

---

## Overview

PipelineRx is an OpenEnv-compliant reinforcement learning environment served over HTTP. An AI agent calls REST endpoints to reset an episode, observe a corrupted pandas DataFrame, apply targeted fixes, and receive a reward signal based on objective data quality criteria.

The environment models a real engineering problem: silent data pipeline failures that cause ML models to degrade in production without obvious errors. The agent must behave like an on-call data engineer, inspecting the DataFrame, identifying root causes, and applying the correct sequence of fixes before the step limit runs out.

The environment is deterministic. Every task generates the same broken data from the same seed and is graded by the same programmatic checks every time. There is no ambiguity and no human evaluator.

---

## Motivation

Data pipeline bugs are among the most common causes of silent ML model failure in production. Issues like type corruption, target leakage, and schema drift are hard to detect automatically, costly to debug manually, and well-suited to agent-based repair. PipelineRx provides a controlled, reproducible environment to train and evaluate agents on exactly these failure modes.

---

## Tasks

PipelineRx has five tasks ranging from easy to hard. Each task generates a broken pandas DataFrame and defines five programmatic sub-tests. The agent earns 0.2 reward per passing sub-test, for a maximum reward of 1.0.

### Task 1: Type Corruption (Easy)

**Dataset:** 2000 rows, 7 columns representing a loan application dataset.

**What is broken:** Numeric columns have been serialized as malformed strings. `interest_rate` contains values like `"8.5%"`. `annual_income` contains values like `"$85,000"`. Null values are stored as the literal string `"missing"` rather than NaN. A model cannot perform arithmetic on strings.

**What the agent must do:** Strip non-numeric characters from each affected column, cast to float64, and replace `"missing"` with NaN.

**Sub-tests:**
1. `interest_rate` is float64 with no percent characters
2. `annual_income` is float64 with no dollar or comma characters
3. `loan_amount` is numeric with no commas
4. `credit_score` is numeric
5. No literal `"missing"` strings remain in any column

**Max steps:** 30

---

### Task 2: Feature Encoding Bug (Easy-Medium)

**Dataset:** 3000 rows, 7 columns representing a customer dataset.

**What is broken:** The `city` column contains 20 cities encoded as integers 1 through 20. This imposes a false ordinal relationship on a nominal variable. The `age` and `monthly_spending` columns are at wildly different scales and have not been normalized. The `signup_date` column is a raw date string rather than a numeric feature.

**What the agent must do:** Remove the false ordinal encoding from `city` (drop or one-hot encode), z-score normalize `age` and `monthly_spending`, and convert `signup_date` to a numeric days-since-epoch value.

**Sub-tests:**
1. `city` is not an ordinal integer column
2. `age` is z-score normalized (mean approximately 0, standard deviation approximately 1)
3. `monthly_spending` is z-score normalized
4. `signup_date` is numeric
5. Protected columns still exist

**Max steps:** 30

---

### Task 3: Schema Drift (Medium)

**Dataset:** Approximately 2700 rows, 6 columns representing a subscription dataset.

**What is broken:** Four simultaneous schema changes have occurred since the model was trained. The column `user_id` was renamed to `customer_id`. A new tier value `"enterprise"` appeared in the `tier` column that was not present during training. Timestamps drifted from UTC to IST (+5.5 hours). Approximately 300 rows were silently dropped.

**What the agent must do:** Rename `customer_id` back to `user_id`, filter out all `"enterprise"` rows, convert timestamps back from IST to UTC, and validate the row count is within the expected range.

**Sub-tests:**
1. Column renamed back to `user_id`
2. No `"enterprise"` values in `tier`
3. Timestamps converted back to UTC range
4. Row count is within the expected range (approximately 2400 to 2600)
5. Core numeric columns still exist

**Max steps:** 30

---

### Task 4: Target Leakage (Hard)

**Dataset:** 4000 rows, 10 columns representing a customer churn dataset.

**What is broken:** The dataset contains three columns that directly leak the prediction target `churned`. The column `days_to_churn` is derived from the future churn date and is only knowable after the event. The column `churn_reason_code` is only assigned after a customer has churned. The column `cancellation_flag` is the target variable stored under a different name. A model trained with these columns will achieve near-perfect training accuracy and near-random production accuracy.

**What the agent must do:** Identify and drop the three leaky columns without touching the five legitimate feature columns (`tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `support_tickets`) and without dropping the target column `churned`.

**Sub-tests:**
1. `days_to_churn` column is removed
2. `churn_reason_code` column is removed
3. `cancellation_flag` column is removed
4. All five legitimate feature columns are preserved
5. Target column `churned` is intact

**Max steps:** 30

---

### Task 5: Cascading Pipeline Failure (Hard)

**Dataset:** A 4-stage data pipeline represented as a set of DataFrames.

**What is broken:** Stage 2 performs a left join that introduces null values in the `revenue` column for unmatched rows. Those nulls propagate through Stage 3's aggregation, corrupting the `total_revenue` totals. Stage 4 double-counts customers due to a COUNT(*) on a non-deduplicated table. The root cause is Stage 2, but the visible symptoms appear in Stages 3 and 4.

**What the agent must do:** Inspect the pipeline stages to trace the root cause, fix Stage 2's join to eliminate nulls, fix Stage 4's double-counting, and produce a clean output DataFrame with valid revenue figures and accurate customer counts.

**Sub-tests:**
1. No null values in `total_revenue`
2. `total_revenue` values are positive and valid
3. `unique_customers` is not double-counted
4. Root cause has been identified as Stage 2
5. All required output columns are present

**Max steps:** 40

---

## Reward Function

Each task is graded by five independent boolean sub-tests. Each passing sub-test contributes 0.2 to the reward, giving a maximum reward of 1.0 and minimum of 0.0.

**Partial credit:** The agent receives reward proportional to how many sub-tests pass at the time `run_tests` is called. This provides a dense signal across the episode rather than a sparse end-of-episode reward.

**Over-fix penalty:** If the agent drops or corrupts a column that was part of a passing sub-test in the previous grader run, the reward for the episode is multiplied by 0.7. This penalizes destructive actions and rewards surgical fixes.

**Time pressure:** After step 20, the reward is reduced by 0.01 per additional step taken. This discourages inefficient exploration and rewards agents that diagnose problems before acting.

**Formula:**

```
reward = (passing_sub_tests / 5.0) * over_fix_multiplier - time_penalty
```

Where `over_fix_multiplier` is 0.7 if a previously passing sub-test is now failing, and 1.0 otherwise. `time_penalty` is `max(0, step_count - 20) * 0.01`.

---

## Observation Space

Observations are plain English descriptions of the current DataFrame state. Each observation includes relevant context for the agent to decide its next action.

After `reset`: task name, difficulty, description, DataFrame shape, column names, max steps, and a diagnostic hint.

After `inspect_schema`: column names, dtypes, and null counts.

After `inspect_data`: a formatted preview of the first N rows.

After `check_column_stats`: descriptive statistics for one column (min, max, mean, std, sample values, null count).

After `apply_fix`: a confirmation message describing what changed, or an error message if the operation failed.

After `run_tests`: each sub-test name, its pass/fail status, and the current total reward.

After `inspect_stage` (Task 5): the shape, column names, dtypes, and sample data for the specified pipeline stage.

---

## Action Space

The agent communicates by sending a JSON object with two keys: `action` (string) and `parameters` (object).

### Read-Only Actions

These actions do not modify the DataFrame. They are free to call at any time.

**inspect_schema**
Returns column names, data types, and null counts for all columns.
```json
{"action": "inspect_schema", "parameters": {}}
```

**inspect_data**
Returns the first N rows of the DataFrame as a formatted string.
```json
{"action": "inspect_data", "parameters": {"n_rows": 5}}
```

**check_column_stats**
Returns descriptive statistics for one column.
```json
{"action": "check_column_stats", "parameters": {"column": "interest_rate"}}
```

**run_tests**
Runs the grader and returns the reward and pass/fail status for each sub-test.
```json
{"action": "run_tests", "parameters": {}}
```

**inspect_stage** (Task 5 only)
Returns information about one pipeline stage.
```json
{"action": "inspect_stage", "parameters": {"stage_id": 2}}
```

### Mutating Actions

These actions modify the DataFrame. All mutating actions go through the `apply_fix` action with an `operation` field.

**strip_and_cast**
Strip specified characters from a string column and cast to a numeric dtype.
```json
{"action": "apply_fix", "parameters": {"column": "interest_rate", "operation": "strip_and_cast", "strip_chars": "%", "target_dtype": "float64"}}
```

**replace_value**
Replace all occurrences of one value in a column with another. Use `"NaN"` as the new value to replace with pandas NaN.
```json
{"action": "apply_fix", "parameters": {"column": "annual_income", "operation": "replace_value", "old_value": "missing", "new_value": "NaN"}}
```

**normalize_zscore**
Z-score normalize a numeric column in place (subtract mean, divide by standard deviation).
```json
{"action": "apply_fix", "parameters": {"column": "age", "operation": "normalize_zscore"}}
```

**normalize_minmax**
Min-max normalize a numeric column to the range [0, 1].
```json
{"action": "apply_fix", "parameters": {"column": "age", "operation": "normalize_minmax"}}
```

**rename_column**
Rename a column.
```json
{"action": "apply_fix", "parameters": {"operation": "rename_column", "old_name": "customer_id", "new_name": "user_id"}}
```

**drop_column**
Drop a column from the DataFrame. Will be rejected if the column is protected.
```json
{"action": "apply_fix", "parameters": {"column": "days_to_churn", "operation": "drop_column"}}
```

**encode_onehot**
One-hot encode a categorical column. The original column is replaced with binary indicator columns.
```json
{"action": "apply_fix", "parameters": {"column": "city", "operation": "encode_onehot"}}
```

**cast_datetime_to_numeric**
Parse a date or datetime column and convert it to days since the Unix epoch (integer).
```json
{"action": "apply_fix", "parameters": {"column": "signup_date", "operation": "cast_datetime_to_numeric"}}
```

**filter_values**
Remove all rows where a column equals a specified value.
```json
{"action": "apply_fix", "parameters": {"column": "tier", "operation": "filter_values", "value": "enterprise"}}
```

**convert_timezone**
Shift a datetime column by a specified number of hours.
```json
{"action": "apply_fix", "parameters": {"column": "created_at", "operation": "convert_timezone", "offset_hours": -5.5}}
```

**fix_stage** (Task 5 only)
Apply a named fix to a pipeline stage.
```json
{"action": "apply_fix", "parameters": {"operation": "fix_stage", "stage_id": 2, "fix_type": "fix_join"}}
```

---

## HTTP API Reference

All endpoints return HTTP 200. Errors are returned in the `observation` field rather than as HTTP error codes, so the agent always receives a valid response.

### GET /health

Returns the server status.

Response:
```json
{"status": "ok"}
```

### GET /tasks

Returns metadata for all five tasks.

Response: array of task objects, each with `id`, `name`, `difficulty`, `description`, `max_steps`, and `hint`.

### POST /reset

Starts a new episode. Generates a fresh broken DataFrame for the specified task.

Request body (all fields optional):
```json
{"task_id": 1, "seed": 42}
```

`task_id` must be between 1 and 5. Default is 1. `seed` controls the random data generation and defaults to 42.

Response:
```json
{
  "task_id": 1,
  "observation": "Task 1: Type Corruption (easy)\n...",
  "info": {
    "tests_passed": 0,
    "tests_total": 5,
    "step_count": 0,
    "task_id": 1,
    "max_steps": 30,
    "warnings": [],
    "session_id": "uuid"
  }
}
```

### POST /step

Executes one action and returns the result.

Request body:
```json
{"action": "inspect_schema", "parameters": {}}
```

Response:
```json
{
  "observation": "DataFrame shape: (2000, 7)\n...",
  "reward": 0.0,
  "done": false,
  "info": {
    "tests_passed": 0,
    "tests_total": 5,
    "step_count": 1,
    "task_id": 1,
    "max_steps": 30,
    "warnings": []
  }
}
```

`reward` is updated only when `run_tests` is called. It retains the last known value between grader calls.

`done` becomes true when all sub-tests pass, the step limit is reached, or the agent calls `run_tests` and achieves a perfect score.

### GET /state

Returns the current episode state without consuming a step.

Response:
```json
{
  "task_id": 1,
  "step_count": 3,
  "reward": 0.4,
  "df_shape": [2000, 7],
  "column_types": {"loan_id": "int64", "interest_rate": "object"},
  "max_steps": 30,
  "done": false
}
```

---

## Pydantic Models

The following typed models define the full OpenEnv interface.

**ResetRequest**
```python
class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=42)
```

**StepRequest**
```python
class StepRequest(BaseModel):
    action: str
    parameters: Dict[str, Any] = {}
```

**StepResponse**
```python
class StepResponse(BaseModel):
    observation: str
    reward: float   # range [0.0, 1.0]
    done: bool
    info: InfoDict
```

**ResetResponse**
```python
class ResetResponse(BaseModel):
    task_id: int
    observation: str
    info: InfoDict
```

**InfoDict**
```python
class InfoDict(BaseModel):
    tests_passed: int
    tests_total: int
    step_count: int
    task_id: int
    max_steps: int
    warnings: List[str]
    root_cause_stage: Optional[int]   # Task 5 only
    session_id: Optional[str]         # returned by /reset
```

**StateResponse**
```python
class StateResponse(BaseModel):
    task_id: int
    step_count: int
    reward: float
    df_shape: List[int]
    column_types: Dict[str, str]
    max_steps: int
    done: bool
```

---

## Baseline Scores

Baseline scores are produced by running `inference.py` with `seed=42` and a GPT-4-class model. The scores reflect the difficulty of each task for a general-purpose LLM without task-specific fine-tuning.

| Task | Name | Difficulty | Baseline Reward |
|------|------|------------|-----------------|
| 1 | Type Corruption | Easy | ~0.90 |
| 2 | Feature Encoding Bug | Easy-Medium | ~0.72 |
| 3 | Schema Drift | Medium | ~0.62 |
| 4 | Target Leakage | Hard | ~0.52 |
| 5 | Cascading Pipeline Failure | Hard | ~0.42 |
| - | Mean | - | ~0.64 |

---

## Setup and Usage

### Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

### Local Installation

```bash
git clone https://github.com/aarushitandon0/pipelinerx
cd pipelinerx
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

The `--workers 1` flag is required. The environment holds a single global state. Multiple workers would each have independent state and would produce inconsistent results.

### Running the Baseline Agent

Set the following environment variables before running `inference.py`:

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your-hugging-face-token"
export ENV_URL="http://localhost:7860"   # optional, defaults to localhost:7860

python inference.py
```

The script runs one episode per task (tasks 1 through 5) and prints structured logs to stdout. It prints a summary of baseline scores at the end.

### Docker

```bash
docker build -t pipelinerx .
docker run -p 7860:7860 pipelinerx
```

The container exposes port 7860. The server starts automatically on container launch.

---

## Inference Script Output Format

`inference.py` emits structured logs to stdout. The format is required by the OpenEnv evaluation harness.

**Episode start:**
```
[START] task_id=1 observation="Task 1: Type Corruption (easy)..."
```

**Each step:**
```
[STEP] task_id=1 step=1 action=inspect_schema reward=0.0000 done=false
[STEP] task_id=1 step=2 action=apply_fix reward=0.2000 done=false
```

**Episode end:**
```
[END] task_id=1 final_reward=1.0000 steps=8 status=success
```

`status` is one of `success` (reward >= 0.95), `partial` (reward >= 0.5), or `fail` (reward < 0.5).

---

## Project Structure

```
pipelinerx/
├── app/
│   ├── __init__.py
│   ├── main.py            FastAPI application, all five HTTP endpoints
│   ├── environment.py     State machine: holds DataFrame, routes actions, computes rewards
│   ├── models.py          Pydantic request/response models (OpenEnv specification)
│   └── tasks/
│       ├── __init__.py    Task registry: GENERATORS, GRADERS, TASK_INFOS dicts
│       ├── task1.py       Type Corruption: generate_data() and grader
│       ├── task2.py       Feature Encoding Bug: generate_data() and grader
│       ├── task3.py       Schema Drift: generate_data() and grader
│       ├── task4.py       Target Leakage: generate_data() and grader
│       └── task5.py       Cascading Pipeline Failure: generate_data() and grader
├── inference.py           Baseline agent script using OpenAI client
├── openenv.yaml           OpenEnv environment metadata
├── Dockerfile             Container definition for Hugging Face Spaces
├── requirements.txt       Python dependencies
├── validate_checklist.py  Pre-submission validation script (58 checks)
├── test_smoke.py          Integration test suite (42 tests)
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier string |
| `HF_TOKEN` | Yes | Hugging Face or API key used as the bearer token |
| `ENV_URL` | No | Base URL of the PipelineRx server. Defaults to `http://localhost:7860` |

---

## Deployment

PipelineRx is deployed to Hugging Face Spaces as a Docker container.

Live URL: `https://areyousheeeee-pipeline-rx.hf.space`

Health check: `https://areyousheeeee-pipeline-rx.hf.space/health`

To deploy your own instance:

1. Create a new Space on Hugging Face with SDK set to Docker.
2. Add the three required secrets (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) in the Space settings under Variables and Secrets.
3. Push the repository to the Space remote. Hugging Face will build and launch the container automatically.
4. The server will be available at `https://your-username-your-space-name.hf.space`.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.111.0 | HTTP API framework |
| uvicorn | 0.30.0 | ASGI server |
| pandas | 2.2.2 | DataFrame operations |
| numpy | 1.26.4 | Synthetic data generation |
| scikit-learn | 1.5.0 | Normalization utilities |
| pydantic | 2.7.1 | Typed request and response models |
| openai | 1.30.0 | LLM client for inference.py |
| requests | 2.32.0 | HTTP client for inference.py |

---

## License

MIT
