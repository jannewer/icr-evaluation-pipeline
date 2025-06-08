import logging

import mlflow
import pandas as pd
from dagster import asset, Output
import icrlearn.rarity as icr_rarity

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.types import DataFrameTuple


@asset(
    description="Rarity scores for the full dataset",
    deps=["preprocessed_dataset"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="rarity_pool",
)
def rarity_scores(
    preprocessed_dataset: tuple[pd.DataFrame, pd.Series],
) -> Output[pd.Series]:
    (X_full, y_full) = preprocessed_dataset

    # Calculate rarity scores for the whole dataset
    # TODO: Use rarity metric(s) depending on measure used in the ICR RF later on
    mlflow.log_param(
        "rarity_metric",
        "CB-LoOP",
    )

    logging.info("Calculating rarity scores for the full dataset")
    rarity_scores = icr_rarity.calculate_cb_loop(X_full, y_full, timing=True)
    rarity_scores = pd.Series(rarity_scores, index=X_full.index)
    logging.info("Finished calculating rarity scores for the full dataset")

    # Log all rarity scores to MLflow as a JSON artifact
    mlflow.log_dict(
        rarity_scores.to_dict(), artifact_file="rarity_scores_full_dataset.json"
    )

    # TODO: Think about useful metadata to include here
    # e.g. the rarest x samples, all samples over a certain rarity threshold, top x percent of rarest samples, etc.
    top_ten_percent_most_rare = rarity_scores.nlargest(int(len(rarity_scores) * 0.1))
    top_10_most_rare = rarity_scores.nlargest(10)
    # Log the top 10 most_rare samples (index and score) as a JSON artifact
    mlflow.log_dict(
        top_10_most_rare.to_dict(),
        artifact_file="top_10_most_rare_samples_full_dataset.json",
    )
    # Log the top 10 percent most_rare samples (index and score) as a JSON artifact
    mlflow.log_dict(
        top_ten_percent_most_rare.to_dict(),
        artifact_file="top_10_percent_most_rare_samples_full_dataset.json",
    )

    return Output(rarity_scores)
