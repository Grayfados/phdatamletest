import pandas as pd
import joblib
import json
import redis
import os
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from


app = Flask(__name__)
swagger = Swagger(app)


try:
    model = joblib.load('model/model.pkl')
    model_features = json.load(open('model/model_features.json'))
    print("Model and features loaded.")
except FileNotFoundError:
    print("Erro: .pkl or feature json not found.")
    model = None
    model_features = None

try:
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print(f"Connected to Redis: {redis_host}")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Redis: {e}")
    redis_client = None



@app.route('/predict', methods=['POST'])
@swag_from('swagger/swagger_doc_predict.yml')
def predict():
    """
    Main endpoint
    """
    if not model or not redis_client:
        return jsonify({"error": "Error starting services."}), 503

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Empty JSON request"}), 400

        house_data_series = pd.Series(input_data)

        zipcode = house_data_series.get('zipcode')
        if not zipcode:
            return jsonify({"error": "'zipcode' is mandatory."}), 400

        demographics_json = redis_client.get(str(int(float(zipcode))))
        if not demographics_json:
            return jsonify({"error": f"Not found demographic data for {zipcode} zipcode."}), 404

        demographics_series = pd.Series(json.loads(demographics_json))

        combined_series = pd.concat([house_data_series, demographics_series])

        final_features_series = combined_series[model_features]

        prediction = model.predict([final_features_series])

        return jsonify({
            "prediction": prediction[0],
            "model_version": "basic_v1",
            "zipcode_used": zipcode
        })

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


@app.route('/predict_basic', methods=['POST'])
@swag_from('swagger/swagger_doc_predict_basic.yml')
def predict_basic():
    """
    Basic endpoint
    """
    if not model or not redis_client:
        return jsonify({"error": "Error starting services."}), 503

    return predict()


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de Health Check para monitoramento.
    """
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"

    model_status = "ok" if model else "error"

    if redis_status == "ok" and model_status == "ok":
        return jsonify({"status": "ok", "redis": redis_status, "model": model_status}), 200
    else:
        return jsonify({"status": "error", "redis": redis_status, "model": model_status}), 503



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)