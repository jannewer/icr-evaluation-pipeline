from dagster import define_asset_job, AssetSelection
from dagster_mlflow import end_mlflow_on_run_finished

from icr_evaluation_pipeline.assets.datasets.datasets import (
    full_dataset,
    k_folds,
    preprocessed_dataset,
)
from icr_evaluation_pipeline.assets.results.individual_results import (
    random_forest_results,
    icr_random_forest_results,
    icr_rf_custom_sampling_results,
)
from icr_evaluation_pipeline.assets.rarity.rarity_scores import rarity_scores
from icr_evaluation_pipeline.assets.results.combined_results import combined_results
from icr_evaluation_pipeline.partitions import dataset_partitions

partitioned_evaluation_job = define_asset_job(
    name="partitioned_model_evaluation",
    partitions_def=dataset_partitions,
    selection=AssetSelection.assets(
        full_dataset,
        preprocessed_dataset,
        rarity_scores,
        k_folds,
        random_forest_results,
        icr_random_forest_results,
        icr_rf_custom_sampling_results,
    ),
    hooks={end_mlflow_on_run_finished},
)

combine_results_job = define_asset_job(
    name="combine_results",
    selection=AssetSelection.assets(combined_results),
    hooks={end_mlflow_on_run_finished},
)
