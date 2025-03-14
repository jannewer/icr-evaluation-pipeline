from dagster import define_asset_job, AssetSelection
from dagster_mlflow import end_mlflow_on_run_finished

from icr_evaluation_pipeline.assets.datasets.datasets import (
    raw_dataset,
    train_and_test_sets,
    test_data_with_rarity_scores,
)
from icr_evaluation_pipeline.assets.models.models import random_forest_model
from icr_evaluation_pipeline.assets.results.individual_results import (
    random_forest_results,
)
from icr_evaluation_pipeline.partitions import dataset_partitions

partitioned_evaluation_job = define_asset_job(
    name="partitioned_model_evaluation",
    partitions_def=dataset_partitions,
    selection=AssetSelection.assets(
        raw_dataset,
        train_and_test_sets,
        test_data_with_rarity_scores,
        random_forest_model,
        random_forest_results,
    ),
    hooks={end_mlflow_on_run_finished},
)
