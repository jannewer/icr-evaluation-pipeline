import logging

import mlflow
import openml
import pandas as pd

from icr_evaluation_pipeline.types import Triple


def encode_categorical_features(
    X: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    # Encode categorical features
    # TODO: Remove completely, if LoOP calculation works without it
    #  (e.g. by specifying a custom similarity matrix based on Gower's distance)
    X[categorical_features] = X[categorical_features].apply(
        lambda col: col.astype("category").cat.codes
    )
    return X


def handle_missing_values(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    col_missing_value_threshold: float,
    row_missing_value_threshold: float,
    full_dataset: openml.OpenMLDataset,
) -> Triple(pd.DataFrame, pd.Series, list[str]):
    X_and_y = pd.concat([X, y], axis=1)

    # Drop columns with over Z% missing values
    columns_to_drop = X.columns[X.isna().mean() > col_missing_value_threshold]
    if not columns_to_drop.empty:
        categorical_features = drop_columns(
            X, X_and_y, categorical_features, columns_to_drop
        )

    number_of_instances = len(X)
    ratio_of_rows_with_missing_values = X.isna().any(axis=1).mean()
    mlflow.log_param(
        "ratio_of_rows_with_missing_values",
        ratio_of_rows_with_missing_values,
    )

    if ratio_of_rows_with_missing_values <= row_missing_value_threshold:
        # Apply CCA (discard rows with missing values)
        X_and_y = X_and_y.dropna()

        # Log the number of rows dropped due to missing values
        mlflow.log_param(
            "num_rows_dropped_due_to_missing_values",
            number_of_instances - len(X_and_y),
        )
        logging.info(
            f"Dataset {full_dataset.name} has been reduced to {len(X_and_y)} instances after applying CCA."
        )
    else:
        # Raise an error if there are too many rows with missing values
        raise ValueError(
            f"Dataset {full_dataset.name} has too many rows with missing values ({ratio_of_rows_with_missing_values:.2%}). "
            "Skipping this dataset."
        )

    y = X_and_y[full_dataset.default_target_attribute]
    X = X_and_y.drop(columns=[full_dataset.default_target_attribute])

    return X, y, categorical_features


def drop_columns(
    X: pd.DataFrame,
    X_and_y: pd.DataFrame,
    categorical_features: list[str],
    columns_to_drop: pd.Index,
) -> list[str]:
    mlflow.log_param(
        "columns_dropped_due_to_missing_ratio_over_threshold",
        columns_to_drop.tolist(),
    )

    X_and_y.drop(columns=columns_to_drop, inplace=True)
    X.drop(columns=columns_to_drop, inplace=True)
    categorical_features = [
        col for col in categorical_features if col not in columns_to_drop.tolist()
    ]
    return categorical_features
