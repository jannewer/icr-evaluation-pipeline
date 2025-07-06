from pathlib import Path
import mlflow
import pandas as pd
from dagster import asset, Output, OpExecutionContext


@asset(
    description="Combined Model Results",
    deps=[
        "random_forest_results",
        "icr_random_forest_results",
        "icr_rf_custom_sampling_results",
    ],
    required_resource_keys={"mlflow"},
    pool="evaluation_pool",
)
def combined_results(
    context: OpExecutionContext,
    icr_random_forest_results: dict[str, tuple[pd.DataFrame, str]],
    random_forest_results: dict[str, tuple[pd.DataFrame, str]],
    icr_rf_custom_sampling_results: dict[str, tuple[pd.DataFrame, str]],
) -> Output[str]:
    """
    Combines the results for all partitions into a pandas dataframe for each model and writes a csv.
    Only includes average values.
    """
    # Get the last backfill run of the evaluation pipeline
    backfills = context.instance.get_backfills(limit=1)
    last_backfill = backfills[0]
    # Get the partitions/datasets included in the last backfill run
    last_backfill_partitions = last_backfill.partition_names
    # Combine the results for each model
    for result in [
        random_forest_results,
        icr_random_forest_results,
        icr_rf_custom_sampling_results,
    ]:
        metrics_list = []
        model_name = "unknown"

        for dataset in result:
            # Skip datasets that were not included in the last backfill run
            if dataset not in last_backfill_partitions:
                continue
            (df, model) = result[dataset]
            model_name = model

            df_dict = df["value"].to_dict()
            df_dict["dataset"] = dataset

            metrics_list.append(df_dict)
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index("dataset", inplace=True)
        # Check if the csv directory exists and create it if it doesn't
        Path("csv").mkdir(exist_ok=True)
        # Create a new directory for every backfill run
        new_directory_path = f"csv/{last_backfill.backfill_id}"
        Path(new_directory_path).mkdir(exist_ok=True)
        # Save the combined results to a csv file and track them in MLflow
        file_name = f"{new_directory_path}/{model_name}_{last_backfill.backfill_id}_combined_results.csv"
        metrics_df.to_csv(file_name)
        mlflow.log_artifact(file_name)
    return Output("Combined results written to combined_results.csv")
