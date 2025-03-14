import mlflow
from dagster import InitResourceContext, ConfigurableResource
from dagster_mlflow.resources import MlFlow


# Overwrite the default MLflow resource to be able to control the experiment name and run name
# Inspired by https://github.com/ion-elgreco/dagster-ml
class mCustomMlflow(type(ConfigurableResource), type(MlFlow)):
    pass


class CustomMlflow(ConfigurableResource, MlFlow, metaclass=mCustomMlflow):
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "Default"
    parent_run_id: str | None = None
    env_tags_to_log: list = []

    class Config:
        frozen = False
        extra = "allow"

    def setup_for_execution(self, context: InitResourceContext) -> None:
        self.log = context.log
        if context.dagster_run is not None:
            # IMPORTANT: Set the mlflow run name to a combination of the partition id and the current dagster run id
            # (except if the job is the combine_results job)
            is_combine_results_job = (
                context.dagster_run.tags.get("dagster/sensor_name")
                == "evaluation_finished_sensor"
            )
            if is_combine_results_job:
                run_name = (
                    f"{context.dagster_run.tags.get('backfill_id')}_combined_results"
                )
            else:
                run_name = f"{context.dagster_run.tags.get('dagster/partition')}_{context.dagster_run.run_id}"
            self.run_name = run_name

            context.log.debug(f"Dagster run tags: {context.dagster_run.tags}")
        else:
            raise ValueError("dagster_run should be available at this point in time.")
        self.dagster_run_id = context.run_id

        self.mlflow_run_id = None
        self.extra_tags = None

        # IMPORTANT: Set the experiment to the custom set backfill_id if the run is related to a previous backfill
        # (i. e. it was triggered by the sensor evaluation_finished_sensor)
        # Otherwise set it to backfill name if the run is part of a backfill, otherwise use the run id
        custom_set_backfill_id = context.dagster_run.tags.get("backfill_id")
        backfill_name = context.dagster_run.tags.get("dagster/backfill")
        run_id = str.split(context.dagster_run.run_id, "-")[0]
        experiment_name = custom_set_backfill_id or backfill_name or run_id
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)

        # Get the client object
        self.tracking_client = mlflow.tracking.MlflowClient()

        # Set up the active run and tags
        self._setup()

    def teardown_after_execution(self, context: InitResourceContext) -> None:
        self.cleanup_on_error()
