import pandas as pd
from dagster import asset, Output
from sklearn.ensemble import RandomForestClassifier

from icr_evaluation_pipeline.evaluation import log_and_persist_metrics
from icr_evaluation_pipeline.partitions import dataset_partitions
from icr_evaluation_pipeline.types import Triple


@asset(
    description="Standard RF Results",
    deps=["test_data_with_rarity_scores", "random_forest_model"],
    partitions_def=dataset_partitions,
    required_resource_keys={"mlflow"},
)
def random_forest_results(
    test_data_with_rarity_scores: Triple(pd.DataFrame, pd.DataFrame, pd.Series),
    random_forest_model: RandomForestClassifier,
) -> Output[tuple[pd.DataFrame, str]]:
    (X_test, y_test, rarity_scores) = test_data_with_rarity_scores
    model_name = "Standard_RF"

    y_pred = random_forest_model.predict(X_test)
    metrics = log_and_persist_metrics(y_test, y_pred, rarity_scores, X_test, model_name)

    return Output((metrics, model_name))
