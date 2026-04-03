# 🏥 PipelineRx

**The gym for ML pipeline debugging.**

An RL environment where AI agents diagnose and repair silent, real-world ML pipeline failures — earning rewards by fixing broken pandas DataFrames across five escalating incident scenarios. No toys. No games. Real data engineering problems.

---

## 🧠 What is PipelineRx?

PipelineRx is an **OpenEnv-compliant reinforcement learning environment** served over HTTP. An AI agent calls REST endpoints to reset an episode, observe a corrupted DataFrame, apply targeted fixes, and receive reward signals. The "world" is a pandas DataFrame in server memory. The agent's job: act like a data engineer at 2am.

### Core Design Principle

PipelineRx sits in the middle — grounded in real tasks (not toys), fully automated with numeric reward (no human evaluator needed), and reproducible (same 5 tasks, same bugs, same grader, every time).

---

## 📋 Five Escalating Incident Scenarios

| # | Task | Difficulty | Description | Expected Reward |
|---|------|------------|-------------|-----------------|
| 1 | **Type Corruption** | Easy | Numeric values stored as malformed strings (`"8.5%"`, `"$85,000"`, `"missing"`) | 0.80 – 1.00 |
| 2 | **Feature Encoding Bug** | Easy-Med | Ordinal encoding of nominal data, un-normalized features, unconverted dates | 0.60 – 0.85 |
| 3 | **Schema Drift** | Medium | Renamed columns, new categories, timezone shifts, dropped rows | 0.50 – 0.75 |
| 4 | **Target Leakage** | Hard | Columns that leak the prediction target (`days_to_churn`, `cancellation_flag`) | 0.40 – 0.70 |
| 5 | **Cascading Pipeline Failure** | Hard | 4-stage pipeline where Stage 2's left join corrupts all downstream stages | 0.30 – 0.60 |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
```

### 3. Test the endpoints

```bash
# Health check
curl http://localhost:7860/health
# → {"status": "ok"}

# List tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Inspect the data
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "inspect_schema", "parameters": {}}'

# Apply a fix
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "apply_fix", "parameters": {"column": "interest_rate", "operation": "strip_and_cast", "strip_chars": "%", "target_dtype": "float64"}}'

# Run tests
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "run_tests", "parameters": {}}'
```

### 4. Run the baseline agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"

python inference.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `POST` | `/reset` | Start new episode. Accepts `{"task_id": 1, "seed": 42}` |
| `POST` | `/step` | Execute one action → observation, reward, done, info |
| `GET` | `/state` | Current task_id, step_count, reward, df_shape, column_types |
| `GET` | `/tasks` | List all 5 tasks with metadata |

### Action Space

**Mutating actions** (via `apply_fix`):

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `strip_and_cast` | Strip chars from strings, cast to dtype | `column`, `strip_chars`, `target_dtype` |
| `replace_value` | Replace one value with another | `column`, `old_value`, `new_value` |
| `normalize_zscore` | Z-score normalize a numeric column | `column` |
| `normalize_minmax` | Min-max normalize to [0,1] | `column` |
| `rename_column` | Rename a column | `old_name`, `new_name` |
| `drop_column` | Drop a column | `column` |
| `encode_onehot` | One-hot encode categorical column | `column` |
| `cast_datetime_to_numeric` | Convert datetime to days-since-epoch | `column` |
| `filter_values` | Remove rows where column equals value | `column`, `value` |
| `convert_timezone` | Shift datetime by UTC offset | `column`, `offset_hours` |
| `fix_stage` | Fix a pipeline stage (Task 5 only) | `stage_id`, `fix_type` |

**Read-only actions**:

| Action | Description | Parameters |
|--------|-------------|------------|
| `inspect_data` | View first N rows | `n_rows` (default 5) |
| `inspect_schema` | Column names, dtypes, nulls | — |
| `run_tests` | Run grader, get reward | — |
| `check_column_stats` | Stats for one column | `column` |
| `inspect_stage` | Pipeline stage info (Task 5) | `stage_id` |

---

## 🎯 Reward Function

- Each task has **5 sub-tests** worth **0.2 each**
- Reward reflects **current data quality** (not cumulative)
- **Partial credit** — agent sees progress after each fix
- **Over-fix penalty**: 30% penalty for dropping protected columns
- **Time pressure**: -0.01 per step over 20

---

## 🐳 Docker Deployment

```bash
docker build -t pipelinerx .
docker run -p 7860:7860 pipelinerx
```

### Deploy to Hugging Face Spaces

1. Create new HF Space → SDK: Docker → Hardware: CPU basic
2. Add secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
3. Push repo. HF auto-builds and maps port 7860.
4. Test: `curl https://your-username-pipelinerx.hf.space/health`

---

## 📁 Project Structure

```
pipeline-rx/
├── app/
│   ├── main.py              ← FastAPI app, all 5 HTTP endpoints
│   ├── environment.py        ← State machine, holds DataFrame, routes actions
│   ├── models.py             ← Pydantic request/response models (OpenEnv spec)
│   └── tasks/
│       ├── __init__.py
│       ├── task1.py          ← Type corruption: generate_data() + grader
│       ├── task2.py          ← Feature encoding: generate_data() + grader
│       ├── task3.py          ← Schema drift: generate_data() + grader
│       ├── task4.py          ← Target leakage: generate_data() + grader
│       └── task5.py          ← Cascading failure: generate_data() + grader
├── inference.py              ← Baseline agent script
├── openenv.yaml              ← OpenEnv metadata
├── Dockerfile                ← Container for HF Spaces
├── requirements.txt
└── README.md
```

---

## 📊 Baseline Scores

Measured with GPT-4o class model, seed=42:

| Task | Difficulty | Expected Reward |
|------|------------|-----------------|
| 1 — Type Corruption | Easy | ~0.90 |
| 2 — Feature Encoding | Easy-Med | ~0.72 |
| 3 — Schema Drift | Medium | ~0.62 |
| 4 — Target Leakage | Hard | ~0.52 |
| 5 — Cascading Failure | Hard | ~0.42 |
| **Mean** | — | **~0.64** |

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face / API key |

---

## 📝 License

MIT
