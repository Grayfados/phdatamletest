import pandas as pd
import requests
import json
import os

API_URL = "http://localhost:5002/invocations"

TEST_FILE_PATH = os.path.join("..", "data", "future_unseen_examples.csv")

N_EXAMPLES = 5
print(f"--- MLflow Model Server Test ---")
print(f"Endpoint: {API_URL}")
print(f"Test Data: {TEST_FILE_PATH}\n")

try:
    if not os.path.exists(TEST_FILE_PATH):
        print(f"Error: Test file not found at '{TEST_FILE_PATH}'")
        print("Please ensure you are running this script from the 'deploy_v2/' directory.")
        exit(1)

    df_test = pd.read_csv(TEST_FILE_PATH)
    test_samples = df_test.head(N_EXAMPLES)
    print(f"Loaded {len(test_samples)} samples for testing.")

except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)
payload_dict = {
    "dataframe_split": test_samples.to_dict(orient='split')
}

print(f"Sending request to {API_URL}...")
try:
    response = requests.post(
        API_URL,
        data=json.dumps(payload_dict),
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print(f"\nSuccess (HTTP {response.status_code}):")
        print("Predictions Received:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nError (HTTP {response.status_code}):")
        print("Response content:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n--- Connection Error ---")
    print(f"Could not connect to {API_URL}.")
    print("Please ensure the 'mlflow models serve' command is running.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")