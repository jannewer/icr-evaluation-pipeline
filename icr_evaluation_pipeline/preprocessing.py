import logging

import mlflow
import numpy as np
import openml
import pandas as pd
from missforest import MissForest

# explicitly require the experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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
    missing_value_threshold: float,
    full_dataset: openml.OpenMLDataset,
) -> Triple(pd.DataFrame, pd.Series, list[str]):
    X_and_y = pd.concat([X, y], axis=1)

    # Drop columns with over Z% missing values
    columns_to_drop = X.columns[X.isna().mean() > missing_value_threshold]
    if not columns_to_drop.empty:
        categorical_features = drop_columns(
            X, X_and_y, categorical_features, columns_to_drop
        )

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
        X_and_y = apply_cca(X_and_y)

        mlflow.log_param("imputation_method", "CCA")
        logging.info(f"Dataset {full_dataset.name} has been imputed with CCA")
    elif 0.1 <= ratio_of_missing_values_X < 0.4:
        if number_of_instances < 1000:
            if more_categorical_than_numerical_features:
                X_and_y = apply_miss_forest(
                    X_and_y,
                    y,
                    categorical_features,
                    target=full_dataset.default_target_attribute,
                )

                # TODO: Remove this after debugging?
                # Log the imputed values to MLflow
                values = [X_and_y.at[row, col] for row, col in missing_indices]
                mlflow.log_param(
                    "imputed_values",
                    values,
                )

                mlflow.log_param("imputation_method", "Miss-Forest")
                logging.info(
                    f"Dataset {full_dataset.name} has been imputed with Miss-Forest"
                )
            else:
                X_and_y = apply_mi(
                    X_and_y, y, target=full_dataset.default_target_attribute
                )

                # TODO: Remove this after debugging?
                # Log the imputed values to MLflow
                values = [X_and_y.at[row, col] for row, col in missing_indices]
                mlflow.log_param(
                    "imputed_values",
                    values,
                )

                mlflow.log_param("imputation_method", "MI")
                logging.info(f"Dataset {full_dataset.name} has been imputed with MI")
        elif 1000 <= number_of_instances < 10000:
            X_and_y = apply_miss_forest(
                X_and_y,
                y,
                categorical_features,
                target=full_dataset.default_target_attribute,
            )

            # TODO: Remove this after debugging?
            # Log the imputed values to MLflow
            values = [X_and_y.at[row, col] for row, col in missing_indices]
            mlflow.log_param(
                "imputed_values",
                values,
            )

            mlflow.log_param("imputation_method", "Miss-Forest")
            logging.info(
                f"Dataset {full_dataset.name} has been imputed with Miss-Forest"
            )
        else:
            X_and_y = apply_cca(X_and_y)

            mlflow.log_param("imputation_method", "CCA")
            logging.info(f"Dataset {full_dataset.name} has been imputed with CCA")
    elif ratio_of_missing_values_X >= 0.4:
        raise ValueError(
            f"Dataset {full_dataset.name} has too many missing values ({ratio_of_missing_values_X:.2%}). "
            "Skipping this dataset."
        )

    mlflow.log_param("shape_after_handling_missing_values", X_and_y.shape)

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


def apply_mi(X_and_y: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    imputer = IterativeImputer()
    X = X_and_y.drop(columns=[target])
    X_and_y = imputer.fit_transform(X)
    X_and_y[target] = y

    return X_and_y


def apply_cca(X_and_y: pd.DataFrame) -> pd.DataFrame:
    # Apply CCA (discard rows with missing values)
    return X_and_y.dropna()


def apply_miss_forest(
    X_and_y: pd.DataFrame, y: pd.Series, categorical_features: list[str], target: str
) -> pd.DataFrame:
    # Encode categorical features with simple Label Encoding for Miss-Forest
    X_and_y[categorical_features] = X_and_y[categorical_features].apply(
        lambda col: col.astype("category").cat.codes
    )

    # Replace -1 with NaN in X_and_y
    X_and_y.replace(-1, np.nan, inplace=True)

    mf = MissForest(categorical=categorical_features)
    X = X_and_y.drop(columns=[target])
    X_and_y = mf.fit_transform(X)
    X_and_y[target] = y

    return X_and_y
