"""Pre-submission checklist verifier for PipelineRx."""
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
checks = []

def check(label, ok):
    checks.append((label, ok))

# 1. Required files exist
for f in ['Dockerfile', 'requirements.txt', 'openenv.yaml', 'inference.py', 'README.md',
          'app/main.py', 'app/environment.py', 'app/models.py', 'app/__init__.py',
          'app/tasks/__init__.py', 'app/tasks/task1.py', 'app/tasks/task2.py',
          'app/tasks/task3.py', 'app/tasks/task4.py', 'app/tasks/task5.py']:
    path = os.path.join(root, f.replace('/', os.sep))
    check(f'File exists: {f}', os.path.isfile(path))

# 2. inference.py at root (not inside subdirectory)
check('inference.py at root', os.path.isfile(os.path.join(root, 'inference.py')))

# 3. openenv.yaml at root
check('openenv.yaml at root', os.path.isfile(os.path.join(root, 'openenv.yaml')))

# 4. Verify imports work
try:
    import app.main
    check('app.main imports', True)
except Exception as e:
    check(f'app.main imports ({e})', False)

# 5. Verify task generators accept seed parameter
from app.tasks import GENERATORS
for tid in range(1, 6):
    try:
        df = GENERATORS[tid](seed=42)
        check(f'Task {tid} generate_data(seed=42) shape={df.shape}', True)
    except Exception as e:
        check(f'Task {tid} generate_data(seed=42) ({e})', False)

# 6. Verify graders return list of 5 bools
from app.tasks import GRADERS
for tid in range(1, 6):
    try:
        df = GENERATORS[tid](seed=42)
        results = GRADERS[tid](df)
        ok = isinstance(results, list) and len(results) == 5 and all(isinstance(r, bool) for r in results)
        check(f'Task {tid} grader returns 5 bools: {results}', ok)
    except Exception as e:
        check(f'Task {tid} grader ({e})', False)

# 7. Verify Pydantic models
try:
    from app.models import (ResetRequest, ResetResponse, StepRequest, StepResponse,
                            StateResponse, TaskInfo, HealthResponse, InfoDict)
    check('Pydantic models import', True)
except Exception as e:
    check(f'Pydantic models ({e})', False)

# 8. Verify Dockerfile contents
with open(os.path.join(root, 'Dockerfile')) as f:
    content = f.read()
    check('Dockerfile has EXPOSE 7860', 'EXPOSE 7860' in content)
    check('Dockerfile has CMD uvicorn', 'uvicorn' in content)
    check('Dockerfile uses python:3.11', 'python:3.11' in content)

# 9. Verify inference.py contents
with open(os.path.join(root, 'inference.py')) as f:
    content = f.read()
    check('inference uses OpenAI client', 'from openai import OpenAI' in content)
    check('inference reads API_BASE_URL', 'API_BASE_URL' in content)
    check('inference reads MODEL_NAME', 'MODEL_NAME' in content)
    check('inference reads HF_TOKEN', 'HF_TOKEN' in content)
    check('inference has [START] format', '[START]' in content)
    check('inference has [STEP] format', '[STEP]' in content)
    check('inference has [END] format', '[END]' in content)

# 10. Verify openenv.yaml has required fields
import yaml  # may not be installed; try plain parsing
try:
    with open(os.path.join(root, 'openenv.yaml')) as f:
        content = f.read()
    for field in ['name', 'description', 'version', 'env', 'tasks', 'action_space', 'observation_space', 'reward']:
        check(f'openenv.yaml has "{field}"', field in content)
except Exception as e:
    check(f'openenv.yaml readable ({e})', False)

# 11. Verify rewards are 0.0-1.0 for broken data (before any fixes)
for tid in range(1, 6):
    df = GENERATORS[tid](seed=42)
    results = GRADERS[tid](df)
    reward = sum(results) / len(results)
    check(f'Task {tid} unfixed reward in [0,1]: {reward:.2f}', 0.0 <= reward <= 1.0)

# 12. Verify 5 tasks listed in tasks/__init__.py
from app.tasks import TASK_INFOS
check(f'TASK_INFOS has 5 entries', len(TASK_INFOS) == 5)
for tid, info in TASK_INFOS.items():
    required = {'id', 'name', 'difficulty', 'description', 'max_steps', 'hint'}
    has_all = required.issubset(info.keys())
    check(f'Task {tid} info has all keys', has_all)

# Print results
passed = sum(1 for _, ok in checks if ok)
total = len(checks)
print("=" * 60)
print("PipelineRx Pre-Submission Checklist")
print("=" * 60)
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
print()
print(f"Results: {passed}/{total}")
if passed == total:
    print("ALL CHECKS PASSED! Ready to submit.")
else:
    print(f"WARNING: {total - passed} checks failed. Fix before submitting.")
print("=" * 60)
sys.exit(0 if passed == total else 1)
