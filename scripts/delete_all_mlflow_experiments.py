import requests

base_url = "http://localhost:5000"
search_url = f"{base_url}/api/2.0/mlflow/experiments/search"
delete_url = f"{base_url}/api/2.0/mlflow/experiments/delete"

# Get all experiments except the default one
search_payload = {
    "max_results": 1000,
    "filter": "name != 'Default'",
}
search_response = requests.post(search_url, json=search_payload)
if search_response.status_code != 200:
    print("Failed to retrieve experiments:", search_response.text)
    exit(1)

experiments_data = search_response.json()
experiments = experiments_data.get("experiments", [])
print(f"Found {len(experiments)} experiment(s).")

# Loop through each experiment and delete it
for exp in experiments:
    experiment_id = exp.get("experiment_id")

    print(f"Deleting experiment with ID: {experiment_id}")
    delete_payload = {"experiment_id": experiment_id}
    delete_response = requests.post(delete_url, json=delete_payload)

    if delete_response.status_code == 200:
        print(f"Successfully deleted experiment {experiment_id}")
    else:
        print(f"Failed to delete experiment {experiment_id}: {delete_response.text}")
