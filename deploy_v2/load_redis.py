import pandas as pd
import redis
import json
import os

print("Loading Feature Store (Redis)...")

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

try:
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    r.ping()
    print(f"Connected to Redis: {REDIS_HOST}")

    df = pd.read_csv('data/zipcode_demographics.csv')
    df = df.set_index('zipcode')

    for zipcode, row in df.iterrows():
        row_json = row.to_json()
        r.set(str(zipcode), row_json)

    print(f"Load finished. {len(df)} registers of zipcode to Redis..")

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)