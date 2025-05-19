import mlflow
import numpy as np
import pandas as pd
import sklearn
from dagster import asset, OpExecutionContext, Output
from icrlearn import ICRRandomForestClassifier
from imblearn.metrics import geometric_mean_score
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from icr_evaluation_pipeline.evaluation import (
    f1_most_rare_score,
    log_and_persist_metrics,
)
from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.types import DataFrameTuple


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    model: sklearn.base.BaseEstimator,
    model_short_name: str,
    rarity_scores: pd.Series,
    dataset_key: str,
) -> pd.DataFrame:
    # TODO: Think about which scoring metrics to use AND which params (especially average)!
    # imbalanced classification report includes: f1, geometric mean, iba, precision, recall, specificity
    scoring = {
        "f1_macro": "f1_macro",
        "geo_macro": make_scorer(
            geometric_mean_score, greater_is_better=True, average="macro"
        ),
        "f1_most_rare_macro": make_scorer(
            f1_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
    }

    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y.squeeze(),
        cv=k_folds,
        scoring=scoring,
        return_indices=True,
        return_estimator=True,
    )

    # Infer the model signature
    example_X = X.iloc[:20, :]
    example_of_fitted_model = cv_results["estimator"][0]
    signature = infer_signature(example_X, example_of_fitted_model.predict(example_X))
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=example_of_fitted_model,
        artifact_path=f"base_models/{dataset_key}",
        signature=signature,
        input_example=example_X,
        registered_model_name=f"base_model_{dataset_key}",
    )

    metrics = log_and_persist_metrics(cv_results, model_short_name)
    return metrics


@asset(
    description="Standard RF Model Results",
    deps=["full_dataset", "k_folds", "rarity_scores"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def random_forest_results(
    context: OpExecutionContext,
    full_dataset: DataFrameTuple,
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    rarity_scores: pd.Series,
) -> Output[tuple[pd.DataFrame, str]]:
    dataset_key = context.partition_key.replace("'", "")
    (X, y) = full_dataset
    base_model = RandomForestClassifier()
    model_short_name = "rf"

    metrics = cross_validate_model(
        X, y, k_folds, base_model, model_short_name, rarity_scores, dataset_key
    )

    return Output((metrics, model_short_name))


@asset(
    description="ICR RF Results",
    deps=["full_dataset", "k_folds"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def icr_random_forest_results(
    context: OpExecutionContext,
    full_dataset: DataFrameTuple,
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    rarity_scores: pd.Series,
) -> Output[tuple[pd.DataFrame, str]]:
    dataset_key = context.partition_key.replace("'", "")
    (X, y) = full_dataset
    icr_model = ICRRandomForestClassifier()
    model_short_name = "icr-rf"

    metrics = cross_validate_model(
        X, y, k_folds, icr_model, model_short_name, rarity_scores, dataset_key
    )

    return Output((metrics, model_short_name))
