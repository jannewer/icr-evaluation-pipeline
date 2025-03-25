import mlflow
import numpy as np
import pandas as pd
from imblearn.metrics import classification_report_imbalanced


def create_and_log_classification_report(
    y_pred: np.ndarray, y_test: pd.DataFrame, suffix: str = "", model_name: str = ""
) -> dict:
    # classification_report_imbalanced returns a dictionary with state-of-the-art metrics for imbalanced classification
    # see https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html for details
    report = classification_report_imbalanced(y_test, y_pred, output_dict=True)
    for key, value in report.items():
        if isinstance(value, dict):
            # Skip logging of metrics for specific classes for now
            # Full metrics are logged as a json file below
            continue
        else:
            # Log all average metrics from the classification report
            key = f"{key}_{model_name}_{suffix}" if suffix else f"{key}_{model_name}"
            mlflow.log_metric(key, value)

    # Log the full metrics dictionary to mlflow (as a json artifact)
    file_name = (
        f"{model_name}_metrics_{suffix}.json"
        if suffix
        else f"{model_name}_metrics.json"
    )
    mlflow.log_dict(report, file_name)

    return report


def log_and_persist_metrics(
    y_test: pd.DataFrame,
    y_pred: np.ndarray,
    rarity_scores: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    # Combine y_pred with the index of X_test to map the predictions back to the original index
    y_pred_with_index = pd.Series(y_pred, index=X_test.index)

    classification_report = create_and_log_classification_report(
        y_pred, y_test, model_name=model_name
    )

    top_10_rarest_samples = rarity_scores.nlargest(10).index.to_numpy()
    classification_report_for_top_10_rarest = create_and_log_classification_report(
        y_pred_with_index[top_10_rarest_samples],
        y_test.loc[top_10_rarest_samples],
        suffix="top_10_rarest",
        model_name=model_name,
    )

    # Create a pandas dataframe from the full metrics dictionaries to return it for further processing
    metrics_df = pd.DataFrame(classification_report).transpose()
    metrics_df.index.name = "label"
    metrics_df_top_10_rarest = pd.DataFrame(
        classification_report_for_top_10_rarest
    ).transpose()
    metrics_df_top_10_rarest.index.name = "label"
    metrics_df_top_10_rarest.index = metrics_df_top_10_rarest.index.map(
        lambda x: f"{x}_top_10_rarest"
    )
    metrics_df = pd.concat([metrics_df, metrics_df_top_10_rarest])

    return metrics_df
