import mlflow
from dagster import asset, OpExecutionContext, Output
from icrlearn import ICRRandomForestClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.types import DataFrameTuple


@asset(
    description="Standard RF Model",
    deps=["training_data"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def random_forest_model(
    context: OpExecutionContext, training_data: DataFrameTuple
) -> Output[RandomForestClassifier]:
    (X_train, y_train) = training_data
    dataset_key = context.partition_key

    base_model = RandomForestClassifier()
    base_model.fit(X_train, y_train)

    # Infer the model signature
    signature = infer_signature(X_train, base_model.predict(X_train))
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=base_model,
        artifact_path=f"base_models/{dataset_key}",
        signature=signature,
        input_example=X_train,
        registered_model_name=f"base_model_{dataset_key}",
    )

    return Output(base_model)


@asset(
    description="ICR RF Model",
    deps=["training_data"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def icr_random_forest_model(
    context: OpExecutionContext, training_data: DataFrameTuple
) -> Output[ICRRandomForestClassifier]:
    (X_train, y_train) = training_data
    dataset_key = context.partition_key

    icr_model = ICRRandomForestClassifier()
    icr_model.fit(X_train, y_train)

    # Infer the model signature
    signature = infer_signature(X_train, icr_model.predict(X_train))
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=icr_model,
        artifact_path=f"icr_models/{dataset_key}",
        signature=signature,
        input_example=X_train,
        registered_model_name=f"icr_model_{dataset_key}",
    )

    return Output(icr_model)
