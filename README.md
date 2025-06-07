# Intra-Class Rarity Model Evaluation Pipeline

This is a pipeline to evaluate intra-class rarity models systematically. It uses the [Dagster](https://dagster.io/)
framework.
[MLflow Tracking](https://mlflow.org/docs/latest/tracking/) is used to log parameters, metrics and artifacts of the
evaluation.

## Getting started

### Prerequisites

Make sure you have the following installed:

- [Python](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Start the Dagster UI web server:

```bash
uv run task dagster
```

Start the MLflow UI web server:

```bash
uv run task mlflow
```

All dependencies will automatically be installed when you run the above commands.
Open http://localhost:3000 with your browser to see the Dagster UI and http://localhost:5000 to see the MLflow UI.

If you want to store Dagster runs across restarts, you can create a `.env` file based on the `.env.example` file and set
the `DAGSTER_HOME` variable to a directory where you want to store the runs.

### Limiting the number of parallel runs

Depending on the number and size of the datasets, the pipeline can take a long time to run and consume a lot of
resources. To avoid overloading your system, you can limit concurrency of the Dagster pipeline.
You can limit the number of parallel runs by adding a `dagster.yaml` file to the directory set as `DAGSTER_HOME` in the
`.env` file. The file should contain the following:

```yaml
concurrency:
  runs:
    max_concurrent_runs: 5
```

More fine-grained options are documented in
the [Dagster documentation](https://docs.dagster.io/guides/operate/managing-concurrency).

### Enabling run monitoring

Dagster can detect hanging runs and restart crashed run workers. To enable this feature, you can add the following to
the `dagster.yaml` file in the directory set as `DAGSTER_HOME`:

```yaml
run_monitoring:
  enabled: true
```

### Setting the OpenML retry policy

If you are using the OpenML repository, you can set the retry policy for OpenML tasks to `robot` as described in
the [OpenML Python User Guide](https://openml.github.io/openml-python/develop/usage.html) by placing a plain text config
file in `~/.config/openml` with the following content:

```
apikey =
server = https://www.openml.org/api/v1/xml
cachedir = /path/to/cachedir
avoid_duplicate_runs = True
connection_n_retries = 50
retry_policy = robot
show_progress = False
```

This will set the retry policy to `robot`, which is suitable for automated tasks and will retry failed tasks more often,
quickly increasing the time between retries.

## Development

### Unit testing

Tests are located in the `icr_evaluation_pipeline_tests` directory and you can run tests using `pytest`:

```bash
pytest icr_evaluation_pipeline_tests
```

### Linting and formatting

Linting and formatting is run automatically on commit using [pre-commit](https://pre-commit.com/).
You can also run it manually using:

```bash
uv run task lint
uv run task format
```

### Adding datasets

[Tasks from the OpenML Repository](https://www.openml.org/search?type=task&sort=runs) can be added by adding their name
to the list of task ids for the partitions definition in `partitions.py`. Adding datasets from other sources will
require modifying the `full_dataset` asset in `datasets.py`.

### Custom MLflow tracking

In order to map one run of the Dagster pipeline (for **all** specified datasets) to one MLflow experiment, the default
MLflow resource is replaced with a custom one. This custom resource is defined in `custom_mlflow_resource.py`.
This way, the experiment name in MLflow is set to the run id of the Dagster pipeline run (the `id` column in the [
`runs` table](http://localhost:3000/runs)).

## Known issues

- The `combine_results` job will fail if not all partitions have been materialized at least one time
