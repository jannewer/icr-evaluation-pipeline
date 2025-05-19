import mlflow
import numpy as np
import pandas as pd
from dagster import (
    asset,
    Output,
    OpExecutionContext,
)
from icrlearn.rarity import calculate_cb_loop
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import StratifiedKFold, KFold
from ucimlrepo import fetch_ucirepo

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import KFoldConfig
from icr_evaluation_pipeline.types import DataFrameTuple, Triple, IteratorType


@asset(
    description="Full dataset",
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def full_dataset(
    context: OpExecutionContext,
) -> Output[DataFrameTuple]:
    dataset_name = context.partition_key
    context.log.info(f"Evaluating model on {dataset_name}")

    # Fetch dataset
    dataset = fetch_ucirepo(name=dataset_name)
    X = dataset.data.features
    y = dataset.data.targets
    dataset_df = pd.concat([X, y], axis=1)

    # Log dataset as input to MLflow
    mlflow_dataset: PandasDataset = mlflow.data.from_pandas(
        dataset_df, name=dataset.metadata.name
    )
    mlflow.log_input(mlflow_dataset, context="training")

    return Output((X, y))


@asset(
    description="K-folds for cross-validation",
    deps=["full_dataset", "rarity_scores"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def k_folds(
    rarity_scores: pd.Series,
    full_dataset: DataFrameTuple,
    config: KFoldConfig,
) -> Output[
    list[tuple[np.ndarray, np.ndarray]]
]:  # TODO: Change to a more specific type if possible
    (X, y) = full_dataset

    if not config.stratify:
        # TODO: Think about whether shuffling is useful here (if it is not sure if some datasets are sorted)
        # Are indices lost when shuffling? We need them for evaluation (top 10 rarest samples, etc.)
        # TODO: Think about the value of k/n_splits
        kf = KFold(n_splits=config.n_splits)
        # Create a list of tuples with the train/test indices
        folds = [
            (train_indices, test_indices) for train_indices, test_indices in kf.split(X)
        ]
    else:
        # TODO: Think about whether shuffling is useful here (if it is not sure if some datasets are sorted)
        # Are indices lost when shuffling? We need them for evaluation (top 10 rarest samples, etc.)
        # TODO: Think about the value of k/n_splits
        skf = StratifiedKFold(n_splits=config.n_splits)

        # Bin the rarity scores to make them categorical (StratifiedKFold requires categorical data for stratification)
        split_criterion = pd.cut(
            rarity_scores, bins=config.n_bins, include_lowest=True, labels=False
        )
        mlflow.log_param("split_criterion_stratified_fold", split_criterion.tolist())

        # Create a list of tuples with the train/test indices
        # The feature to split on is sufficient to determine the split
        # Therefore np.zeros(n_samples) may be used as a placeholder for X instead of actual training data
        folds = [
            (train_indices, test_indices)
            for train_indices, test_indices in skf.split(
                X=np.zeros(X.shape[0]), y=split_criterion
            )
        ]

    # Log the k-folds and params to MLflow
    mlflow.log_param("n_splits", config.n_splits)
    mlflow.log_param("stratify", config.stratify)
    mlflow.log_param("n_bins", config.n_bins)
    for i, (train_indices, test_indices) in enumerate(folds):
        mlflow.log_param(f"fold_{i}_indices_train", train_indices.tolist())
        mlflow.log_param(f"fold_{i}_indices_test", test_indices.tolist())

    return Output(folds)
