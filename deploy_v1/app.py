import pandas as pd
import joblib
import json
import redis
import os
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

# --- Inicialização da Aplicação ---

app = Flask(__name__)
# Configuração do Flasgger (Swagger)
swagger = Swagger(app)

# --- Carregamento de Modelos e Features ---

try:
    # Carrega o modelo serializado
    model = joblib.load('model/model.pkl')
    # Carrega a lista de features que o modelo espera, na ordem correta
    model_features = json.load(open('model/model_features.json'))
    print("Modelo e lista de features carregados com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivos 'model.pkl' ou 'model_features.json' não encontrados.")
    model = None
    model_features = None

# --- Conexão com o Feature Store (Redis) ---

try:
    # Obtém o host do Redis a partir da variável de ambiente definida no docker-compose
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print(f"Conectado ao Redis em: {redis_host}")
except redis.exceptions.ConnectionError as e:
    print(f"Erro ao conectar ao Redis: {e}")
    redis_client = None


# --- Endpoints da API ---

@app.route('/predict', methods=['POST'])
@swag_from('swagger/swagger_doc_predict.yml')  # (Vamos criar este yml depois, por enquanto é só um placeholder)
def predict():
    """
    Endpoint principal de predição.
    Recebe todos os dados do formulário da casa [cite: 58] e enriquece
    com dados demográficos do Redis  antes de fazer a predição.
    """
    if not model or not redis_client:
        return jsonify({"error": "Serviço não inicializado corretamente."}), 503

    try:
        # 1. Pega o JSON de entrada
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Requisição JSON vazia."}), 400

        # Converte para Pandas Series para facilitar a manipulação
        house_data_series = pd.Series(input_data)

        zipcode = house_data_series.get('zipcode')
        if not zipcode:
            return jsonify({"error": "O campo 'zipcode' é obrigatório."}), 400

        # 2. Busca dados demográficos no Redis
        demographics_json = redis_client.get(str(int(float(zipcode))))
        if not demographics_json:
            return jsonify({"error": f"Dados demográficos para o zipcode {zipcode} não encontrados."}), 404

        demographics_series = pd.Series(json.loads(demographics_json))

        # 3. Combina as fontes de dados
        combined_series = pd.concat([house_data_series, demographics_series])

        # 4. Filtra e ordena as features para o modelo
        # Isso garante que estamos passando os dados na ordem exata que o modelo espera
        final_features_series = combined_series[model_features]

        # 5. Faz a predição
        # O modelo espera uma lista 2D, por isso [final_features_series]
        prediction = model.predict([final_features_series])

        # 6. Retorna o resultado
        return jsonify({
            "prediction": prediction[0],
            "model_version": "basic_v1",
            "zipcode_used": zipcode
        })

    except Exception as e:
        return jsonify({"error": f"Erro durante a predição: {str(e)}"}), 500


@app.route('/predict_basic', methods=['POST'])
@swag_from('swagger/swagger_doc_predict_basic.yml')  # (Placeholder)
def predict_basic():
    """
    [BÔNUS] Endpoint de predição básico.
    Recebe *apenas* as features necessárias da casa [cite: 68] (além do zipcode
    para a consulta) e faz a predição.
    """
    if not model or not redis_client:
        return jsonify({"error": "Serviço não inicializado corretamente."}), 503

    # A lógica é idêntica à do /predict, pois o 'model_features'
    # já filtra automaticamente apenas o que é necessário.
    # A diferença é que o usuário pode enviar um JSON menor.

    # Podemos reutilizar a função de predição:
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


# --- Execução Local (para testes) ---

if __name__ == '__main__':
    # Roda o app em modo de debug se executado diretamente
    app.run(debug=True, host='0.0.0.0', port=5000)