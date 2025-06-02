import numpy as np
import openml
from dagster import StaticPartitionsDefinition

benchmark_suite = openml.study.get_suite("OpenML-CC18")
all_task_ids = np.array(benchmark_suite.tasks).astype("str").tolist()
task_ids_to_remove = ["2079"]
selected_task_ids = [
    task_id for task_id in all_task_ids if task_id not in task_ids_to_remove
]

dataset_partitions = StaticPartitionsDefinition(selected_task_ids)
