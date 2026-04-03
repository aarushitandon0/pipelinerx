"""Task registry."""

from app.tasks.task1 import generate_data as generate_task1, grade as grade_task1, TASK_INFO as TASK1_INFO
from app.tasks.task2 import generate_data as generate_task2, grade as grade_task2, TASK_INFO as TASK2_INFO
from app.tasks.task3 import generate_data as generate_task3, grade as grade_task3, TASK_INFO as TASK3_INFO
from app.tasks.task4 import generate_data as generate_task4, grade as grade_task4, TASK_INFO as TASK4_INFO
from app.tasks.task5 import generate_data as generate_task5, grade as grade_task5, TASK_INFO as TASK5_INFO

GENERATORS = {
    1: generate_task1,
    2: generate_task2,
    3: generate_task3,
    4: generate_task4,
    5: generate_task5,
}

GRADERS = {
    1: grade_task1,
    2: grade_task2,
    3: grade_task3,
    4: grade_task4,
    5: grade_task5,
}

TASK_INFOS = {
    1: TASK1_INFO,
    2: TASK2_INFO,
    3: TASK3_INFO,
    4: TASK4_INFO,
    5: TASK5_INFO,
}
