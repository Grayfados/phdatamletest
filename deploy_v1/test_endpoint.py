import pandas as pd
import requests
import json

# URL do nosso serviço rodando no Docker
API_URL = "http://localhost:5000/predict"

# Arquivo com os exemplos de teste
TEST_FILE = "data/future_unseen_examples.csv"

# Número de exemplos que queremos testar
N_EXAMPLES = 5

print(f"Starting tests: {API_URL}")
print(f"Reading from: {TEST_FILE}\n")

try:
    df_test = pd.read_csv(TEST_FILE)
    test_samples = df_test.head(N_EXAMPLES)

    for index, row in test_samples.iterrows():
        payload = row.to_dict()
        payload.update({'zipcode': str(int(payload['zipcode']))})

        print(f"--- Sending sample {index} (zipcode: {payload.get('zipcode')}) ---")

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                print(f"Success (HTTP {response.status_code}):")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error (HTTP {response.status_code}):")
                print(response.text)

        except requests.exceptions.ConnectionError as ce:
            print(f"Connection error: {ce}")
            break

        print("-" * (20 + len(str(index))))

except FileNotFoundError:
    print(f"ERROR: '{TEST_FILE}' not found.")
except Exception as e:
    print(f"ERROR: {e}")