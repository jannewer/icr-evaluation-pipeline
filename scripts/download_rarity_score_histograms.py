import os
import argparse
from mlflow.tracking import MlflowClient
import requests

ARTIFACT_PATH_DEFAULT = "rarity_scores_histogram.png"


def download_histograms(
    base_url: str,
    experiment_name: str,
    output_dir: str,
    artifact_path: str,
    username: str = None,
    password: str = None,
) -> None:
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        auth = (username, password)
    else:
        auth = None

    client = MlflowClient(tracking_uri=base_url)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(
            f"Error: Experiment with name '{experiment_name}' not found at {base_url}."
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    for run in runs:
        run_id = run.info.run_id
        run_name = run.info.run_name
        dataset_id = run_name.split("_")[0] if "_" in run_name else run_name
        print(f"Processing run {run_id} for dataset {dataset_id}")
        url = f"{base_url}/get-artifact?path={artifact_path}&run_id={run_id}"
        output_path = os.path.join(output_dir, f"{dataset_id}_{artifact_path}")

        try:
            response = requests.get(url, auth=auth, timeout=30)
            response.raise_for_status()
            with open(output_path, "wb") as out_file:
                out_file.write(response.content)
            print(f"Downloaded artifact for run {run_id} to {output_path}")
        except requests.exceptions.HTTPError as http_err:
            print(
                f"Failed to download artifact for run {run_id}: {http_err} (status code: {response.status_code})"
            )
        except Exception as e:
            print(f"Failed to download artifact for run {run_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all rarity score histograms for an MLflow experiment."
    )
    parser.add_argument(
        "--base-url",
        "-b",
        required=True,
        help="Base URL of the MLflow tracking server (e.g. http://localhost:5000)",
    )
    parser.add_argument(
        "--experiment-name", "-e", required=True, help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="rarity_score_histograms",
        help="Directory to save downloaded artifacts",
    )
    parser.add_argument(
        "--artifact-path",
        "-a",
        default=ARTIFACT_PATH_DEFAULT,
        help=f"Relative artifact path in Mlflow (default: {ARTIFACT_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--username",
        "-u",
        help="Username for basic auth (if configured on remote host)",
    )
    parser.add_argument(
        "--password",
        "-p",
        help="Password for basic auth (if configured on remote host)",
    )
    args = parser.parse_args()

    download_histograms(
        base_url=args.base_url,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        artifact_path=args.artifact_path,
        username=args.username,
        password=args.password,
    )


if __name__ == "__main__":
    main()
