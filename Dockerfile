FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

COPY federated-learning-v2/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY federated-learning-v2/ .

RUN echo "hello"
