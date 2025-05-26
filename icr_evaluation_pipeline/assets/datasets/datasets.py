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
from sklearn.model_selection import StratifiedKFold, KFold

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import KFoldConfig
from icr_evaluation_pipeline.types import DataFrameTuple


@asset(
    description="Full dataset",
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
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
)
def preprocessed_dataset(
    full_dataset: openml.OpenMLDataset,
) -> Output[DataFrameTuple]:
    X, y, categorical_indicator, _ = full_dataset.get_data(
        target=full_dataset.default_target_attribute, dataset_format="dataframe"
    )
    X_and_y = pd.concat([X, y], axis=1)

    number_of_instances = len(X)
    number_of_categorical_features = np.array(categorical_indicator).sum()
    number_of_numerical_features = X.shape[1] - number_of_categorical_features
    more_categorical_than_numerical_features = (
        number_of_categorical_features > number_of_numerical_features
    )
    ratio_of_missing_values_X = X.isnull().sum() / len(X)

    # For features with missing values do the following:
    # - If the ratio of missing values is < 0.1 apply CCA
    # - If the ratio of missing values is >= 0.1 and < 0.4:
    # - If the number of instances in the dataset is < 1000:
    # - If there are more categorical features than numerical features apply Miss-Forest
    # - If there are more numerical features than categorical features apply MI
    # - If the number of instances in the dataset is >= 1000 and < 10000 apply Miss-Forest
    # - If the number of instances in the dataset is >= 10000 apply CCA
    # TODO: If the ratio of missing values is >= 0.5 apply Mixed Methods?

    features_with_missing_values_less_than_10_percent = ratio_of_missing_values_X[
        (ratio_of_missing_values_X > 0) & (ratio_of_missing_values_X < 0.1)
    ]
    features_with_missing_values_between_10_and_40_percent = ratio_of_missing_values_X[
        (ratio_of_missing_values_X >= 0.1) & (ratio_of_missing_values_X < 0.4)
    ]
    features_with_missing_values_greater_than_40_percent = ratio_of_missing_values_X[
        ratio_of_missing_values_X >= 0.4
    ]

    # Log the feature names and their missing value ratios to MLflow
    mlflow.log_param(
        "features_with_missing_values_less_than_10_percent",
        features_with_missing_values_less_than_10_percent.index.tolist(),
    )
    mlflow.log_param(
        "features_with_missing_values_between_10_and_40_percent",
        features_with_missing_values_between_10_and_40_percent.index.tolist(),
    )
    mlflow.log_param(
        "features_with_missing_values_greater_than_40_percent",
        features_with_missing_values_greater_than_40_percent.index.tolist(),
    )

    for feature in features_with_missing_values_less_than_10_percent.index:
        print(
            f"Feature {feature} has {ratio_of_missing_values_X[feature]:.2%} missing values"
        )

        # Apply CCA (discard rows with missing values)
        X_and_y.dropna(subset=[feature], inplace=True)

        print(f"Feature {feature} has been imputed with CCA")

    for feature in features_with_missing_values_between_10_and_40_percent.index:
        if number_of_instances < 1000:
            if more_categorical_than_numerical_features:
                # TODO: Apply Miss-Forest
                pass
            else:
                # TODO: Apply MI
                pass
        elif 1000 <= number_of_instances < 10000:
            # TODO: Apply Miss-Forest
            pass
        else:
            # Apply CCA (discard rows with missing values)
            X_and_y.dropna(subset=[feature], inplace=True)

    # TODO: for every feature in missing_values_greater_than_40_percent apply Mixed Methods

    y = X_and_y[full_dataset.default_target_attribute]
    # TODO: Adjust the other steps instead --> Make them use a Series instead of a DataFrame for y
    y = pd.DataFrame(y)
    X = X_and_y.drop(columns=[full_dataset.default_target_attribute])

    # Update the index of X and y to match after dropping rows
    # TODO: Is this necessary?
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Encode categorical features
    # TODO: Use label encoding instead! Or even remove completely, if LoOP calculation works without it
    #  (e.g. by specifying a custom similarity matrix based on Gower's distance)
    # One-hot encoding (int is necessary for LoOP calculation)
    X = pd.get_dummies(X, drop_first=True, dtype=np.int8)

    return Output((X, y))


@asset(
    description="K-folds for cross-validation",
    deps=["preprocessed_dataset", "rarity_scores"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def k_folds(
    rarity_scores: pd.Series,
    preprocessed_dataset: DataFrameTuple,
    config: KFoldConfig,
) -> Output[list[tuple[np.ndarray, np.ndarray]]]:
    (X, y) = preprocessed_dataset

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
