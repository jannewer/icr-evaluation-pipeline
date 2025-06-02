import mlflow
import numpy as np
import openml
import pandas as pd
from dagster import (
    asset,
    Output,
    OpExecutionContext,
)
from mlflow.data.pandas_dataset import PandasDataset

# explicitly require the experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import StratifiedKFold, KFold

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.preprocessing import (
    handle_missing_values,
    encode_categorical_features,
)
from icr_evaluation_pipeline.resources.configs import KFoldConfig, PreprocessingConfig


@asset(
    description="Full dataset",
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="preprocessing_pool",
)
def full_dataset(
    context: OpExecutionContext,
) -> Output[openml.OpenMLDataset]:
    dataset_id = int(context.partition_key)
    openml_task = openml.tasks.get_task(dataset_id)

    dataset_name = openml_task.get_dataset().name
    context.log.info(f"Evaluating model on {dataset_name}")

    # Fetch dataset from OpenML
    dataset = openml.datasets.get_dataset(openml_task.dataset_id)

    # Log dataset as input to MLflow
    X, y, _, _ = openml_task.get_dataset().get_data(
        target=openml_task.target_name, dataset_format="dataframe"
    )
    dataset_df = pd.concat([X, y], axis=1)
    mlflow_dataset: PandasDataset = mlflow.data.from_pandas(
        dataset_df, name=dataset_name
    )
    mlflow.log_input(mlflow_dataset, context="training")

    return Output(dataset)


@asset(
    description="Preprocessed dataset",
    deps=["full_dataset"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="preprocessing_pool",
)
def preprocessed_dataset(
    full_dataset: openml.OpenMLDataset, config: PreprocessingConfig
) -> Output[tuple[pd.DataFrame, pd.Series]]:
    X, y, categorical_indicator, _ = full_dataset.get_data(
        target=full_dataset.default_target_attribute, dataset_format="dataframe"
    )
    categorical_features = X.columns[categorical_indicator].tolist()

    try:
        X, y, categorical_features = handle_missing_values(
            X,
            y,
            categorical_features,
            config.col_missing_value_threshold,
            config.row_missing_value_threshold,
            full_dataset,
        )
    except ValueError as ve:
        mlflow.log_param("error_handling_missing_values", str(ve))
        raise ve

    X = encode_categorical_features(X, categorical_features)

    return Output((X, y))


@asset(
    description="K-folds for cross-validation",
    deps=["preprocessed_dataset", "rarity_scores"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="preprocessing_pool",
)
def k_folds(
    rarity_scores: pd.Series,
    preprocessed_dataset: tuple[pd.DataFrame, pd.Series],
    config: KFoldConfig,
) -> Output[list[tuple[np.ndarray, np.ndarray]]]:
    (X, _) = preprocessed_dataset

    if not config.stratify:
        kf = KFold(n_splits=config.n_splits, shuffle=config.shuffle)
        # Create a list of tuples with the train/test indices
        folds = [
            (train_indices, test_indices) for train_indices, test_indices in kf.split(X)
        ]
    else:
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.shuffle)

        # Bin the rarity scores to make them categorical (StratifiedKFold requires categorical data for stratification)
        split_criterion = pd.cut(
            rarity_scores, bins=config.n_bins, include_lowest=True, labels=False
        )
        mlflow.log_param("split_criterion_stratified_k_fold", split_criterion.tolist())
        mlflow.log_param("n_bins_stratified_k_fold", config.n_bins)

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
    mlflow.log_param("n_splits_k_fold", config.n_splits)
    mlflow.log_param("stratified_k_fold", config.stratify)

    # Log full folds to MLflow as a JSON artifact
    folds_as_dict = {
        f"fold_{i}": {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        }
        for i, (train_indices, test_indices) in enumerate(folds)
    }
    mlflow.log_dict(folds_as_dict, artifact_file="k_folds.json")

    return Output(folds)
