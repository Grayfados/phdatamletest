import pandas as pd
import requests
import json

# URL do nosso serviço rodando no Docker
API_URL = "http://localhost:5000/predict"

# Arquivo com os exemplos de teste
TEST_FILE = "data/future_unseen_examples.csv"

# Número de exemplos que queremos testar
N_EXAMPLES = 5

print(f"Iniciando teste do endpoint: {API_URL}")
print(f"Lendo dados de: {TEST_FILE}\n")

try:
    # Carrega os dados de exemplo
    df_test = pd.read_csv(TEST_FILE)

    # Pega os primeiros N exemplos
    test_samples = df_test.head(N_EXAMPLES)

    # Itera sobre cada amostra e envia para a API
    for index, row in test_samples.iterrows():
        # Converte a linha do pandas para um dicionário (formato JSON)
        payload = row.to_dict()
        payload.update({'zipcode': str(int(payload['zipcode']))})

        print(f"--- Enviando Amostra {index} (zipcode: {payload.get('zipcode')}) ---")

        try:
            # Envia a requisição POST com o payload em JSON
            response = requests.post(API_URL, json=payload)

            # Verifica se a requisição foi bem-sucedida
            if response.status_code == 200:
                print(f"Sucesso (HTTP {response.status_code}):")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Erro (HTTP {response.status_code}):")
                print(response.text)

        except requests.exceptions.ConnectionError:
            print("Erro de Conexão: Não foi possível conectar à API.")
            print("Verifique se os containers (docker-compose) estão rodando.")
            break

        print("-" * (20 + len(str(index))))

except FileNotFoundError:
    print(f"Erro: Arquivo de teste '{TEST_FILE}' não encontrado.")
except Exception as e:
    print(f"Um erro inesperado ocorreu: {e}")