import pandas as pd
import redis
import json
import os

print("Iniciando carregamento do Feature Store (Redis)...")

# O nome 'redis' funciona aqui porque é o nome do serviço no docker-compose.yml
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

try:
    # Conecta ao Redis
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    r.ping()
    print(f"Conectado ao Redis em: {REDIS_HOST}")

    # Carrega os dados demográficos
    df = pd.read_csv('data/zipcode_demographics.csv')
    df = df.set_index('zipcode')

    # Itera e salva no Redis (chave = zipcode, valor = JSON da linha)
    for zipcode, row in df.iterrows():
        # Converte a linha (que é uma Series) para JSON
        row_json = row.to_json()
        r.set(str(zipcode), row_json)

    print(f"Carga concluída. {len(df)} registros de CEP carregados no Redis.")

except Exception as e:
    print(f"Erro ao conectar ou carregar dados no Redis: {e}")
    # Sai com erro para que o docker-compose saiba que falhou
    exit(1)