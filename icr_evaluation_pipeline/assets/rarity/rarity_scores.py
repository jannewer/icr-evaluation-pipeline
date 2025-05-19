import mlflow
import pandas as pd
from dagster import asset, Output
from icrlearn.rarity import calculate_cb_loop

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.types import DataFrameTuple


@asset(
    description="Rarity scores for the full dataset",
    deps=["full_dataset"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def rarity_scores(
    full_dataset: DataFrameTuple,
) -> Output[pd.Series]:
    (X_full, y_full) = full_dataset

    # Calculate rarity scores for the whole dataset
    # TODO: Use rarity metric(s) depending on measure used in the ICR RF later on
    rarity_scores = calculate_cb_loop(X_full, y_full)
    rarity_scores = pd.Series(rarity_scores, index=X_full.index)

    # TODO: Think about useful metadata to include here
    # e.g. the rarest x samples, all samples over a certain rarity threshold, top x percent of rarest samples, etc.
    top_10_rarest = rarity_scores.nlargest(10).index.tolist()
    mlflow.log_param("top_10_rarest_samples", top_10_rarest)

    return Output(rarity_scores)
