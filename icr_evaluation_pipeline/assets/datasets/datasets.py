import mlflow
import numpy as np
import pandas as pd
from dagster import (
    asset,
    Output,
    OpExecutionContext,
    multi_asset,
    AssetOut,
)
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.resources.configs import TrainAndTestDatasetConfig
from icr_evaluation_pipeline.types import DataFrameTuple, Triple


@asset(
    description="Raw dataset",
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def raw_dataset(
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


@multi_asset(
    outs={
        "training_data": AssetOut(description="Train dataset"),
        "test_data": AssetOut(description="Test dataset"),
    },
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def train_and_test_sets(
    raw_dataset: DataFrameTuple,
    config: TrainAndTestDatasetConfig,
) -> tuple[Output[DataFrameTuple], Output[DataFrameTuple]]:
    (X, y) = raw_dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=config.seed, train_size=config.train_size
    )

    # Log train/test split parameters to MLflow
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    return (
        Output((X_train, y_train)),
        Output((X_test, y_test)),
    )


@asset(
    description="Test dataset with rarity scores",
    deps=["test_data"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def test_data_with_rarity_scores(
    test_data: DataFrameTuple,
) -> Output[Triple(pd.DataFrame, pd.DataFrame, pd.Series)]:
    (X_test, y_test) = test_data

    # TODO: Use proper rarity metric(s) here
    # For now, we'll use random scores as a placeholder
    rarity_scores = np.random.rand(len(X_test))
    rarity_scores = pd.Series(rarity_scores, index=X_test.index)

    # TODO: Think about useful metadata to include here
    # e.g. the rarest x samples, all samples over a certain rarity threshold, top x percent of rarest samples, etc.
    top_10_rarest = rarity_scores.nlargest(10).index.tolist()

    mlflow.log_param("top_10_rarest_samples", top_10_rarest)

    return Output((X_test, y_test, rarity_scores))
