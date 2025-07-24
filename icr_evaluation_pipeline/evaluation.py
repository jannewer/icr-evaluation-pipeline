import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def get_y_most_rare(
    y_true: pd.Series, y_pred: npt.ArrayLike, rarity_scores: pd.Series
) -> tuple[npt.NDArray, npt.NDArray]:
    # Get the rarity scores for all samples in the test set of the current fold
    rarity_scores_for_y_true = rarity_scores.loc[y_true.index]

    # Get the indices of the top 10 percent rarest samples of the test set
    top_ten_percent_most_rare_indices = rarity_scores_for_y_true.nlargest(
        int(len(rarity_scores_for_y_true) * 0.1)
    ).index

    # Create a series from y_pred with the same index as y_true to be able to locate the rarest samples by index
    y_pred_series = pd.Series(y_pred, index=y_true.index)
    # Create numpy arrays that contain the actual and predicted values for the rarest samples
    y_true_most_rare = y_true.loc[top_ten_percent_most_rare_indices].to_numpy()
    y_pred_most_rare = y_pred_series.loc[top_ten_percent_most_rare_indices].to_numpy()
    return y_pred_most_rare, y_true_most_rare


def accuracy_most_rare_score(
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    rarity_scores: pd.Series,
    sample_weight: np.ndarray = None,
) -> float:
    y_pred_most_rare, y_true_most_rare = get_y_most_rare(y_true, y_pred, rarity_scores)

    # Calculate accuracy score for the rarest samples
    return accuracy_score(
        y_true_most_rare, y_pred_most_rare, sample_weight=sample_weight
    )


def precision_most_rare_score(
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    rarity_scores: pd.Series,
    average: str = "macro",
    sample_weight: np.ndarray = None,
    pos_label: int | float | bool | str = None,
) -> float:
    y_pred_most_rare, y_true_most_rare = get_y_most_rare(y_true, y_pred, rarity_scores)

    # Calculate precision score for the rarest samples
    return precision_score(
        y_true_most_rare,
        y_pred_most_rare,
        average=average,
        sample_weight=sample_weight,
        pos_label=pos_label,  # ignored if average != "binary"
    )


def recall_most_rare_score(
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    rarity_scores: pd.Series,
    average: str = "macro",
    sample_weight: np.ndarray = None,
    pos_label: int | float | bool | str = None,
) -> float:
    y_pred_most_rare, y_true_most_rare = get_y_most_rare(y_true, y_pred, rarity_scores)

    # Calculate recall score for the rarest samples
    return recall_score(
        y_true_most_rare,
        y_pred_most_rare,
        average=average,
        sample_weight=sample_weight,
        pos_label=pos_label,  # ignored if average != "binary"
    )


def f1_most_rare_score(
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    rarity_scores: pd.Series,
    average: str = "macro",
    sample_weight: np.ndarray = None,
    pos_label: int | float | bool | str = None,
) -> float:
    y_pred_most_rare, y_true_most_rare = get_y_most_rare(y_true, y_pred, rarity_scores)

    # Calculate f1 score for the rarest samples
    return f1_score(
        y_true_most_rare,
        y_pred_most_rare,
        average=average,
        sample_weight=sample_weight,
        pos_label=pos_label,  # ignored if average != "binary"
    )


def specificity_most_rare_score(
    y_true: pd.Series,
    y_pred: npt.ArrayLike,
    rarity_scores: pd.Series,
    average: str = "macro",
    sample_weight: np.ndarray = None,
    pos_label: int | float | bool | str = None,
) -> float:
    y_pred_most_rare, y_true_most_rare = get_y_most_rare(y_true, y_pred, rarity_scores)

    # Calculate specificity score for the rarest samples
    return specificity_score(
        y_true_most_rare,
        y_pred_most_rare,
        average=average,
        sample_weight=sample_weight,
        pos_label=pos_label,  # ignored if average != "binary"
    )


def log_and_persist_metrics(
    cv_results: dict[str, npt.NDArray],
    model_name: str,
) -> pd.DataFrame:
    # Log full cross validation results to mlflow (as a json artifact)
    file_name = f"{model_name}_cv_results.json"
    mlflow.log_dict(cv_results, file_name)

    metrics_dict = {}
    # Log metrics for all folds
    for key, value in cv_results.items():
        # Log mean and std for all scoring metrics and fit and score time
        if (
            key.startswith("test_")
            or key.startswith("fit_")
            or key.startswith("score_")
        ):
            key = key.replace("test_", "")
            std_key = f"std_{key}_{model_name}"
            mean_key = f"mean_{key}_{model_name}"
            metrics_dict[mean_key] = np.mean(value, dtype=np.float64)
            metrics_dict[std_key] = np.std(value, dtype=np.float64)
            mlflow.log_metric(mean_key, np.mean(value, dtype=np.float64))
            mlflow.log_metric(std_key, np.std(value, dtype=np.float64))

    # Return all mean and std metrics as a dataframe for further processing
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    metrics_df.columns = ["value"]

    return metrics_df
