# Usamos uma imagem base slim para manter o tamanho reduzido
FROM python:3.10-slim

# Definimos o diretório de trabalho dentro do container
WORKDIR /app

# Copiamos primeiro o requirements.txt para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copia todo o resto do projeto (app.py, model/, data/, etc.)
COPY . .

# Expõe a porta que o Gunicorn vai usar
EXPOSE 5000

# Comando para rodar a aplicação com Gunicorn
# 4 workers, bind em todas as interfaces na porta 5000, app:app (arquivo:objeto)
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]