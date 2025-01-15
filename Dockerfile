# Base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /flib

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    openjdk-11-jdk \
    python3.10 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Maven
RUN wget https://downloads.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz -O - | tar xzf - -C /usr/share && \
    ln -s /usr/share/apache-maven-3.9.6 /usr/share/maven && \
    ln -s /usr/share/maven/bin/mvn /usr/bin/mvn    

# Set the default Python version to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Java dependencies and setup AMLsim
COPY flib/sim flib/sim
WORKDIR /flib/flib/sim/AMLsim
RUN mvn install:install-file \
    -Dfile=jars/mason.20.jar \
    -DgroupId=mason \
    -DartifactId=mason \
    -Dversion=20 \
    -Dpackaging=jar \
    -DgeneratePom=true
RUN mvn clean package -DskipTests
RUN sh scripts/run.sh

# Copy the rest of the files
WORKDIR /flib
COPY flib/preprocess flib/preprocess
COPY flib/train flib/train
COPY flib/tune flib/tune
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
COPY setup.cfg .
COPY setup.py .

# Install flib
RUN pip3 install --no-cache-dir -e .

# Copy the examples and set as working directory
COPY experiments experiments
WORKDIR /flib/experiments

ENTRYPOINT ["bash"]