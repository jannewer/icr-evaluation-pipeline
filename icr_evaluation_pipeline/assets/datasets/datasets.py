import logging

import mlflow
import numpy as np
import openml
import pandas as pd
from dagster import (
    asset,
    Output,
    OpExecutionContext,
)
from missforest import MissForest
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import StratifiedKFold, KFold

# explicitly require the experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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
    categorical_features = X.columns[categorical_indicator].tolist()

    # Drop columns with over 50% missing values
    columns_to_drop = X.columns[X.isna().mean() > 0.5]
    if not columns_to_drop.empty:
        mlflow.log_param(
            "columns_dropped_due_to_missing_ratio_over_50_percent",
            columns_to_drop.tolist(),
        )
        X_and_y.drop(columns=columns_to_drop, inplace=True)
        X.drop(columns=columns_to_drop, inplace=True)
        categorical_features = [
            col for col in categorical_features if col not in columns_to_drop.tolist()
        ]

    # For the whole dataset do the following:
    # - If the ratio of missing values is < 0.1 apply CCA
    # - If the ratio of missing values is >= 0.1 and < 0.4:
    # - If the number of instances in the dataset is < 1000:
    # - If there are more categorical features than numerical features apply Miss-Forest
    # - If there are more numerical features than categorical features apply MI
    # - If the number of instances in the dataset is >= 1000 and < 10000 apply Miss-Forest
    # - If the number of instances in the dataset is >= 10000 apply CCA
    # TODO: If the ratio of missing values is >= 0.5 apply Mixed Methods?
    # TODO: OR Escalate this to the user as the dataset might be a bad fir for evaluation?

    number_of_instances = len(X)
    number_of_categorical_features = len(categorical_features)
    number_of_numerical_features = X.shape[1] - number_of_categorical_features
    more_categorical_than_numerical_features = (
        number_of_categorical_features > number_of_numerical_features
    )
    ratio_of_missing_values_X = X.isna().values.mean()
    mlflow.log_param("ratio_of_missing_values", ratio_of_missing_values_X)
    mlflow.log_param("shape_before_handling_missing_values", X_and_y.shape)

    # TODO: Remove this after debugging?
    # Log cell indices (row, column) of missing values in X
    missing_indices = X.isna().stack()[X.isna().stack()].index.tolist()
    mlflow.log_param("missing_indices", missing_indices)

    if 0 < ratio_of_missing_values_X < 0.1:
        # Apply CCA (discard rows with missing values)
        X_and_y.dropna(inplace=True)
        mlflow.log_param("imputation_method", "CCA")
        print(f"Dataset {full_dataset.name} has been imputed with CCA")
    elif 0.1 <= ratio_of_missing_values_X < 0.4:
        if number_of_instances < 1000:
            if more_categorical_than_numerical_features:
                # Dataset 125920

                # Encode categorical features with simple Label Encoding for Miss-Forest
                X_and_y[categorical_features] = X_and_y[categorical_features].apply(
                    lambda col: col.astype("category").cat.codes
                )
                # Replace -1 with NaN in X_and_y
                X_and_y.replace(-1, np.nan, inplace=True)

                mf = MissForest(categorical=categorical_features)
                X_and_y = mf.fit_transform(
                    X_and_y.drop(columns=[full_dataset.default_target_attribute])
                )
                X_and_y[full_dataset.default_target_attribute] = y

                mlflow.log_param("imputation_method", "Miss-Forest")
                logging.info(
                    f"Dataset {full_dataset.name} has been imputed with Miss-Forest"
                )
            else:
                # TODO: Apply MI instead (e.g. with sklearn's IterativeImputer or with MICEForest)
                # Dataset: Eucalyptus (2079) (???)

                imputer = IterativeImputer(random_state=42)
                X_and_y = imputer.fit_transform(
                    X_and_y.drop(columns=[full_dataset.default_target_attribute])
                )
                X_and_y[full_dataset.default_target_attribute] = y

                mlflow.log_param("imputation_method", "MI")
                logging.info(f"Dataset {full_dataset.name} has been imputed with MI")
        elif 1000 <= number_of_instances < 10000:
            # Encode categorical features with simple Label Encoding for Miss-Forest
            X_and_y[categorical_features] = X_and_y[categorical_features].apply(
                lambda col: col.astype("category").cat.codes
            )
            # Replace -1 with NaN in X_and_y
            X_and_y.replace(-1, np.nan, inplace=True)

            mf = MissForest(categorical=categorical_features)
            X_and_y = mf.fit_transform(
                X_and_y.drop(columns=[full_dataset.default_target_attribute])
            )
            X_and_y[full_dataset.default_target_attribute] = y

            mlflow.log_param("imputation_method", "Miss-Forest")
            logging.info(
                f"Dataset {full_dataset.name} has been imputed with Miss-Forest"
            )
        else:
            # Apply CCA (discard rows with missing values)
            categorical_features = X.columns[categorical_indicator].tolist()

            # Encode categorical features with simple Label Encoding for Miss-Forest
            X[categorical_features] = X[categorical_features].apply(
                lambda col: col.astype("category").cat.codes
            )

            mf = MissForest(categorical=categorical_features)
            X_and_y = mf.fit_transform(X_and_y)

            mlflow.log_param("imputation_method", "Miss-Forest")
            logging.info(
                f"Dataset {full_dataset.name} has been imputed with Miss-Forest"
            )

            # X_and_y.dropna(inplace=True)
            # mlflow.log_param("imputation_method", "CCA")
            # logging.info(f"Dataset {full_dataset.name} has been imputed with CCA")
    elif ratio_of_missing_values_X >= 0.4:
        # TODO: Think about escalating this, as this might not be a good dataset for the evaluation
        X_and_y.dropna(inplace=True)
        mlflow.log_param("imputation_method", "Mixed-Methods")
        logging.info(f"Dataset {full_dataset.name} has been imputed with Mixed-Methods")

    mlflow.log_param("shape_after_handling_missing_values", X_and_y.shape)

    # TODO: Remove this after debugging?
    # Log dataset after handling missing values to MLflow
    mlflow_dataset_after_imputation: PandasDataset = mlflow.data.from_pandas(
        X_and_y, name=f"{full_dataset.name}_after_imputation"
    )
    mlflow.log_input(mlflow_dataset_after_imputation, context="imputation")

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
