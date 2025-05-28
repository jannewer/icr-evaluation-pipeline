import numpy as np
import openml
from dagster import StaticPartitionsDefinition

# OpenMl Partition
benchmark_suite = openml.study.get_suite("OpenML-CC18")
task_ids = np.array(benchmark_suite.tasks).astype("str").tolist()
dataset_partitions = StaticPartitionsDefinition(task_ids)
