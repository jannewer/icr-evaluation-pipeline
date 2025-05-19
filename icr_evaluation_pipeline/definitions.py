from dagster import Definitions, load_assets_from_package_module

from icr_evaluation_pipeline.assets import datasets, results, rarity
from icr_evaluation_pipeline.jobs import partitioned_evaluation_job, combine_results_job
from icr_evaluation_pipeline.resources.custom_mlflow import CustomMlflow
from icr_evaluation_pipeline.sensors.evaluation_finished import (
    evaluation_finished_sensor,
)

dataset_assets = load_assets_from_package_module(datasets, group_name="Datasets")
rarity_assets = load_assets_from_package_module(rarity, group_name="Rarity_Scores")
result_assets = load_assets_from_package_module(
    results, group_name="Cross_Validation_Results"
)

all_assets = [
    *dataset_assets,
    *rarity_assets,
    *result_assets,
]

all_resources = {
    "mlflow": CustomMlflow(),
}

all_jobs = [partitioned_evaluation_job, combine_results_job]

all_sensors = [evaluation_finished_sensor]

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    resources=all_resources,
    sensors=all_sensors,
)
