import mlflow
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
from icr_evaluation_pipeline.types import DataFrameTuple


@asset(
    description="Raw dataset",
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def raw_dataset(
    context: OpExecutionContext,
) -> Output[DataFrameTuple]:
    # TODO: Is this still necessary?
    mlflow.set_tag("backfill", context.run.tags.get("dagster/backfill"))

    dataset_name = context.partition_key
    print(f"Evaluating model on {dataset_name}")

    # fetch dataset
    dataset = fetch_ucirepo(name=dataset_name)
    X = dataset.data.features
    y = dataset.data.targets
    dataset_df = pd.concat([X, y], axis=1)

    # log dataset as input to MLflow
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

    # log train/test split parameters to MLflow
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    return (
        Output((X_train, y_train)),
        Output((X_test, y_test)),
    )
