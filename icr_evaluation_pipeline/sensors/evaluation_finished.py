from typing import Iterator

from dagster import sensor, RunRequest, DefaultSensorStatus, SensorEvaluationContext
from dagster._core.execution.backfill import BulkActionStatus

from icr_evaluation_pipeline.jobs import combine_results_job


@sensor(job=combine_results_job, default_status=DefaultSensorStatus.RUNNING)
def evaluation_finished_sensor(
    context: SensorEvaluationContext,
) -> Iterator[RunRequest]:
    """
    Triggers the combine_results job whenever a backfill run of the evaluation pipeline is successfully completed
    (e.g. when the partitioned_model_evaluation job is finished for all selected datasets).
    """
    # Get the last backfill run of the evaluation pipeline
    backfills = context.instance.get_backfills(limit=1)
    last_backfill = backfills[0] if backfills else None
    # Check if the last backfill was successful and is not still running
    if last_backfill and last_backfill.status == BulkActionStatus.COMPLETED_SUCCESS:
        # Set the run key to the last backfill id --> This will trigger the combine_results job ONCE
        # When the sensor is triggered again, the last backfill id will be the same and the job will not be triggered again
        key = f"{last_backfill.backfill_id}_combined_results"
        yield RunRequest(run_key=key, tags={"backfill_id": last_backfill.backfill_id})
    else:
        context.log.info(
            "Skipping combine_results job because the last backfill was not successful or is still running"
        )
