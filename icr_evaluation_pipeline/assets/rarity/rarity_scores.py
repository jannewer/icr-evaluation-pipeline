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
    preprocessed_dataset: DataFrameTuple,
) -> Output[pd.Series]:
    (X_full, y_full) = preprocessed_dataset

    # Log columns with missing values to MLflow
    missing_values = X_full.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index.tolist()
    mlflow.log_param("missing_columns", missing_columns)

    # Calculate rarity scores for the whole dataset
    # TODO: Use rarity metric(s) depending on measure used in the ICR RF later on
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
    top_10_rarest = rarity_scores.nlargest(10)
    mlflow.log_param("top_10_rarest_samples_full_dataset", top_10_rarest.index.tolist())
    # Log the top 10 rarest samples (index and score) as a JSON artifact
    mlflow.log_dict(
        top_10_rarest.to_dict(), artifact_file="top_10_rarest_samples_full_dataset.json"
    )

    return Output(rarity_scores)
