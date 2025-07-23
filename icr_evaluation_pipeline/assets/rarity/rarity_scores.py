import logging

import icrlearn.rarity as icr_rarity
import mlflow
import pandas as pd
from dagster import asset, Output

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import RarityScoreConfig


@asset(
    description="Rarity scores for the full dataset",
    deps=["preprocessed_dataset"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="rarity_pool",
)
def rarity_scores(
    preprocessed_dataset: tuple[pd.DataFrame, pd.Series], config: RarityScoreConfig
) -> Output[pd.Series]:
    (X_full, y_full) = preprocessed_dataset

    # Calculate rarity scores for the whole dataset
    mlflow.log_param(
        "rarity_measure_rarity_scores",
        config.rarity_measure,
    )
    logging.info("Calculating rarity scores for the full dataset")
    if config.rarity_measure == "cb_loop":
        if config.n_neighbors is None:
            rarity_scores = icr_rarity.calculate_cb_loop(
                X_full,
                y_full,
                min_score=config.min_rarity_score,
                extent=config.cb_loop_extent,
            )
        else:
            rarity_scores = icr_rarity.calculate_cb_loop(
                X_full,
                y_full,
                min_score=config.min_rarity_score,
                extent=config.cb_loop_extent,
                n_neighbors=config.n_neighbors,
            )
        rarity_scores = pd.Series(rarity_scores, index=X_full.index)
    elif config.rarity_measure == "l2class":
        if config.n_neighbors is None:
            rarity_scores = icr_rarity.calculate_l2class(
                X_full,
                y_full,
                psi=config.l2class_psi,
                beta=config.min_rarity_score,
            )
        else:
            rarity_scores = icr_rarity.calculate_l2class(
                X_full,
                y_full,
                n_neighbors=config.n_neighbors,
                psi=config.l2class_psi,
                beta=config.min_rarity_score,
            )
        rarity_scores = pd.Series(rarity_scores, index=X_full.index)
    else:
        raise ValueError(
            f"Unsupported rarity measure: {config.rarity_measure}. "
            "Supported measures are 'cb_loop' and 'l2class'."
        )
    logging.info("Finished calculating rarity scores for the full dataset")

    # Log all rarity scores to MLflow as a JSON artifact
    mlflow.log_dict(
        rarity_scores.to_dict(), artifact_file="rarity_scores_full_dataset.json"
    )

    # Log the histogram of rarity scores to MLflow
    rarity_scores_histogram = rarity_scores.hist(bins=100, figsize=(10, 6), log=True)
    mlflow.log_figure(
        figure=rarity_scores_histogram.get_figure(),
        artifact_file="rarity_scores_histogram.png",
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
