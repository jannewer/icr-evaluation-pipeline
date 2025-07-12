import logging

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icr_evaluation_pipeline.evaluation import (
    f1_most_rare_score,
    log_and_persist_metrics,
    geo_most_rare_score,
)
from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import ModelConfig
from icr_evaluation_pipeline.types import DataFrameTuple


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
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
        "geo_most_rare_macro": make_scorer(
            geo_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
    }

    logging.info(
        f"Starting cross-validation of {model_short_name} model on dataset {dataset_key}"
    )

    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=k_folds,
        scoring=scoring,
        return_indices=True,
        return_estimator=True,
    )

    logging.info(
        f"Finished cross-validation of {model_short_name} model on dataset {dataset_key}"
    )

    # Infer the model signature
    example_X = X.iloc[:20, :]
    example_of_fitted_model = cv_results["estimator"][0]
    signature = infer_signature(example_X, example_of_fitted_model.predict(example_X))
    # Log the model
    artifact_path = (
        f"base_models/{dataset_key}"
        if model_short_name == "rf"
        else f"icr_models/{dataset_key}"
    )
    model_name = (
        f"base_model_{dataset_key}"
        if model_short_name == "rf"
        else f"icr_model_{dataset_key}"
    )
    mlflow.sklearn.log_model(
        sk_model=example_of_fitted_model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=example_X,
        registered_model_name=model_name,
    )

    metrics = log_and_persist_metrics(cv_results, model_short_name)
    return metrics


@asset(
    description="Standard RF Model Results",
    deps=["preprocessed_dataset", "k_folds", "rarity_scores"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="evaluation_pool",
)
def random_forest_results(
    context: OpExecutionContext,
    preprocessed_dataset: tuple[pd.DataFrame, pd.Series],
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    rarity_scores: pd.Series,
) -> Output[tuple[pd.DataFrame, str]]:
    # TODO: This is the id not the name when using OpenML --> Think about how to handle this
    dataset_key = context.partition_key.replace("'", "")
    (X, y) = preprocessed_dataset
    base_model = RandomForestClassifier(n_jobs=-1)
    model_short_name = "rf"

    metrics = cross_validate_model(
        X, y, k_folds, base_model, model_short_name, rarity_scores, dataset_key
    )

    return Output((metrics, model_short_name))


@asset(
    description="ICR RF Results",
    deps=["preprocessed_dataset", "k_folds"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
    pool="evaluation_pool",
)
def icr_random_forest_results(
    context: OpExecutionContext,
    preprocessed_dataset: tuple[pd.DataFrame, pd.Series],
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    rarity_scores: pd.Series,
    config: ModelConfig,
) -> Output[tuple[pd.DataFrame, str]]:
    dataset_key = context.partition_key.replace("'", "")
    (X, y) = preprocessed_dataset

    mlflow.log_param("rarity_adjustment_method", config.rarity_adjustment_method)
    mlflow.log_param("rarity_measure_model_evaluation", config.rarity_measure)
    mlflow.log_param("n_neighbors", config.n_neighbors)
    mlflow.log_param("min_rarity_score", config.min_rarity_score)
    mlflow.log_param("cb_loop_extent", config.cb_loop_extent)
    mlflow.log_param("l2min_psi", config.l2min_psi)

    icr_model = ICRRandomForestClassifier(
        n_jobs=-1,
        rarity_adjustment_method=config.rarity_adjustment_method,
        rarity_measure=config.rarity_measure,
        n_neighbors=config.n_neighbors,
        min_rarity_score=config.min_rarity_score,
        cb_loop_extent=config.cb_loop_extent,
        l2min_psi=config.l2min_psi,
    )
    model_short_name = "icr-rf"

    metrics = cross_validate_model(
        X, y, k_folds, icr_model, model_short_name, rarity_scores, dataset_key
    )

    return Output((metrics, model_short_name))
