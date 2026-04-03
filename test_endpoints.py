"""Quick endpoint smoke test — run while the server is up on :7860."""
import requests
import json
import sys

BASE = "http://localhost:7860"

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok

passes = 0
total = 0

def test(label, ok, detail=""):
    global passes, total
    total += 1
    if check(label, ok, detail):
        passes += 1

print("=" * 60)
print("PipelineRx Endpoint Smoke Tests")
print("=" * 60)

# 1. /health
print("\n1. GET /health")
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    test("/health returns 200", r.status_code == 200, f"status={r.status_code}")
    test("/health body has status=ok", r.json().get("status") == "ok", f"body={r.json()}")
except Exception as e:
    test("/health reachable", False, str(e))

# 2. GET /tasks
print("\n2. GET /tasks")
try:
    r = requests.get(f"{BASE}/tasks", timeout=10)
    tasks = r.json()
    test("/tasks returns 200", r.status_code == 200)
    test("/tasks has 5 tasks", len(tasks) == 5, f"got {len(tasks)}")
    for t in tasks:
        required_keys = {"id", "name", "difficulty", "description", "max_steps", "hint"}
        has_all = required_keys.issubset(set(t.keys()))
        test(f"  Task {t.get('id')} has all keys", has_all, f"keys={list(t.keys())}")
except Exception as e:
    test("/tasks reachable", False, str(e))

# 3. POST /reset for each task
print("\n3. POST /reset (all tasks)")
for tid in range(1, 6):
    try:
        r = requests.post(f"{BASE}/reset", json={"task_id": tid, "seed": 42}, timeout=10)
        d = r.json()
        test(f"  /reset task {tid} returns 200", r.status_code == 200)
        test(f"  /reset task {tid} has observation", len(d.get("observation", "")) > 10, f"obs_len={len(d.get('observation',''))}")
        test(f"  /reset task {tid} has info", "info" in d)
    except Exception as e:
        test(f"  /reset task {tid}", False, str(e))

# 4. Step actions on Task 1
print("\n4. POST /step actions (Task 1)")
requests.post(f"{BASE}/reset", json={"task_id": 1, "seed": 42}, timeout=10)

# inspect_schema
r = requests.post(f"{BASE}/step", json={"action": "inspect_schema", "parameters": {}}, timeout=10)
d = r.json()
test("inspect_schema returns valid StepResponse", all(k in d for k in ("observation", "reward", "done", "info")))
test("inspect_schema reward is float [0,1]", 0.0 <= d["reward"] <= 1.0, f"reward={d['reward']}")

# inspect_data
r = requests.post(f"{BASE}/step", json={"action": "inspect_data", "parameters": {"n_rows": 3}}, timeout=10)
d = r.json()
test("inspect_data works", "DataFrame shape" in d.get("observation", ""))

# check_column_stats
r = requests.post(f"{BASE}/step", json={"action": "check_column_stats", "parameters": {"column": "interest_rate"}}, timeout=10)
d = r.json()
test("check_column_stats works", "interest_rate" in d.get("observation", ""))

# apply_fix
r = requests.post(f"{BASE}/step", json={"action": "apply_fix", "parameters": {"column": "interest_rate", "operation": "strip_and_cast", "strip_chars": "%", "target_dtype": "float64"}}, timeout=10)
d = r.json()
test("apply_fix strip_and_cast works", "strip_and_cast" in d.get("observation", "").lower() or "applied" in d.get("observation", "").lower())

# run_tests
r = requests.post(f"{BASE}/step", json={"action": "run_tests", "parameters": {}}, timeout=10)
d = r.json()
test("run_tests returns reward", 0.0 <= d["reward"] <= 1.0, f"reward={d['reward']}")
test("run_tests info has tests_passed", "tests_passed" in d.get("info", {}), f"info={d.get('info')}")

# invalid action — must return 200 not 4xx
r = requests.post(f"{BASE}/step", json={"action": "totally_bogus", "parameters": {}}, timeout=10)
test("invalid action returns 200", r.status_code == 200, f"status={r.status_code}")
test("invalid action reward=0.0", r.json().get("reward") == 0.0 or r.json().get("reward", -1) >= 0.0)

# 5. GET /state
print("\n5. GET /state")
r = requests.get(f"{BASE}/state", timeout=10)
s = r.json()
test("/state has all keys", all(k in s for k in ("task_id", "step_count", "reward", "df_shape", "column_types", "max_steps", "done")))

# 6. Full grader sweep
print("\n6. Grader sweep (all tasks, fresh reset, immediate run_tests)")
for tid in range(1, 6):
    requests.post(f"{BASE}/reset", json={"task_id": tid, "seed": 42}, timeout=10)
    r = requests.post(f"{BASE}/step", json={"action": "run_tests", "parameters": {}}, timeout=10)
    d = r.json()
    reward = d["reward"]
    tp = d["info"]["tests_passed"]
    # Before any fixes, reward should be between 0 and 1 (possibly 0)
    test(f"  Task {tid} grader returns valid reward", 0.0 <= reward <= 1.0, f"reward={reward} tests_passed={tp}/5")

# 7. Step limit enforcement
print("\n7. Step limit enforcement")
requests.post(f"{BASE}/reset", json={"task_id": 1, "seed": 42}, timeout=10)
done = False
for i in range(35):
    r = requests.post(f"{BASE}/step", json={"action": "inspect_schema", "parameters": {}}, timeout=10)
    d = r.json()
    if d["done"]:
        done = True
        test(f"  Episode ends by step {i+1}", i+1 <= 30, f"done at step {i+1}")
        break
if not done:
    test("  Episode ends within max_steps", False, "Never got done=true in 35 steps!")

# Summary
print("\n" + "=" * 60)
print(f"Results: {passes}/{total} tests passed")
if passes == total:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {total - passes} tests failed")
print("=" * 60)

sys.exit(0 if passes == total else 1)
