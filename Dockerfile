FROM python:3.10-slim

WORKDIR /app

COPY federated-learning-v2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY federated-learning-v2/ .

# CMD ["python", "main.py"]