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

Open http://localhost:3000 with your browser to see the Dagster UI and http://localhost:5000 to see the MLflow UI.

If you want to store Dagster runs across restarts, you can create a `.env` file based on the `.env.example` file and set
the `DAGSTER_HOME` variable to a directory where you want to store the runs.

## Development

### Unit testing

Tests are in the `icr_evaluation_pipeline_tests` directory and you can run tests using `pytest`:

```bash
pytest icr_evaluation_pipeline_tests
```
