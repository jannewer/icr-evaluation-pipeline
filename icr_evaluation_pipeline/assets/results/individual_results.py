import logging

import mlflow
import numpy as np
import openml
import pandas as pd
import sklearn
from dagster import asset, OpExecutionContext, Output
from icrlearn import ICRRandomForestClassifier
from imblearn.metrics import specificity_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from icr_evaluation_pipeline.evaluation import (
    f1_most_rare_score,
    log_and_persist_metrics,
    accuracy_most_rare_score,
    precision_most_rare_score,
    recall_most_rare_score,
    specificity_most_rare_score,
)
from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import ModelConfig


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: list[tuple[np.ndarray, np.ndarray]],
    model: sklearn.base.BaseEstimator,
    model_short_name: str,
    rarity_scores: pd.Series,
    dataset_key: str,
) -> pd.DataFrame:
    dataset_task = openml.tasks.get_task(int(dataset_key))
    dataset_name = dataset_task.get_dataset().name

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "precision_macro": "precision_macro",
        "precision_micro": "precision_micro",
        "recall_macro": "recall_macro",
        "recall_micro": "recall_micro",
        "specificity_macro": make_scorer(
            specificity_score, greater_is_better=True, average="macro"
        ),
        "specificity_micro": make_scorer(
            specificity_score, greater_is_better=True, average="micro"
        ),
        "accuracy_most_rare": make_scorer(
            accuracy_most_rare_score,
            greater_is_better=True,
            rarity_scores=rarity_scores,
        ),
        "f1_most_rare_macro": make_scorer(
            f1_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "f1_most_rare_micro": make_scorer(
            f1_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "precision_most_rare_macro": make_scorer(
            precision_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "precision_most_rare_micro": make_scorer(
            precision_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "recall_most_rare_macro": make_scorer(
            recall_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "recall_most_rare_micro": make_scorer(
            recall_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "specificity_most_rare_macro": make_scorer(
            specificity_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "specificity_most_rare_micro": make_scorer(
            specificity_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
    }

    logging.info(
        f"Starting cross-validation of {model_short_name} model on dataset {dataset_name} (key: {dataset_key})"
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
        f"Finished cross-validation of {model_short_name} model on dataset {dataset_name} (key: {dataset_key})"
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
    (X, y) = preprocessed_dataset
    base_model = RandomForestClassifier(n_jobs=-1)
    model_short_name = "rf"

    metrics = cross_validate_model(
        X,
        y,
        k_folds=k_folds,
        model=base_model,
        model_short_name=model_short_name,
        rarity_scores=rarity_scores,
        dataset_key=context.partition_key,
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
    (X, y) = preprocessed_dataset

    mlflow.log_param("rarity_adjustment_method", config.rarity_adjustment_method)
    mlflow.log_param("rarity_measure_model_evaluation", config.rarity_measure)
    mlflow.log_param("n_neighbors", config.n_neighbors)
    mlflow.log_param("min_rarity_score", config.min_rarity_score)
    mlflow.log_param("cb_loop_extent", config.cb_loop_extent)
    mlflow.log_param("l2class_psi", config.l2class_psi)

    icr_model = ICRRandomForestClassifier(
        n_jobs=-1,
        rarity_adjustment_method=config.rarity_adjustment_method,
        rarity_measure=config.rarity_measure,
        n_neighbors=config.n_neighbors,
        min_rarity_score=config.min_rarity_score,
        cb_loop_extent=config.cb_loop_extent,
        l2class_psi=config.l2class_psi,
    )
    model_short_name = "icr-rf"

    metrics = cross_validate_model(
        X,
        y,
        k_folds=k_folds,
        model=icr_model,
        model_short_name=model_short_name,
        rarity_scores=rarity_scores,
        dataset_key=context.partition_key,
    )

    return Output((metrics, model_short_name))
