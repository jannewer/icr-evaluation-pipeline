from dagster import Definitions, load_assets_from_package_module

from icr_evaluation_pipeline.assets import datasets, models, results
from icr_evaluation_pipeline.resources.custom_mlflow import CustomMlflow

dataset_assets = load_assets_from_package_module(datasets, group_name="Datasets")
model_assets = load_assets_from_package_module(models, group_name="Trained_Models")
result_assets = load_assets_from_package_module(results, group_name="Test_Results")

all_assets = [
    *dataset_assets,
    *model_assets,
    *result_assets,
]

all_resources = {
    "mlflow": CustomMlflow(),
}

defs = Definitions(
    assets=all_assets,
    resources=all_resources,
)
