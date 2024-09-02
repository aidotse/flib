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

# Install java dependencies
COPY AMLsim/jars AMLsim/jars
RUN mvn install:install-file \
    -Dfile=AMLsim/jars/mason.20.jar \
    -DgroupId=mason \
    -DartifactId=mason \
    -Dversion=20 \
    -Dpackaging=jar \
    -DgeneratePom=true
    
# Set the default Python version to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Setup AMLsim
WORKDIR /flib/AMLsim
COPY AMLsim/scripts scripts
COPY AMLsim/src src
COPY AMLsim/pom.xml pom.xml
RUN mvn clean package -DskipTests
RUN sh scripts/run.sh

# Setup preprocess
WORKDIR /flib
COPY preprocess/ preprocess/

# Setup auto-aml-data-gen
WORKDIR /flib/auto-aml-data-gen
COPY auto-aml-data-gen/classifier.py classifier.py 
COPY auto-aml-data-gen/main.py main.py
COPY auto-aml-data-gen/optimizer.py optimizer.py
COPY auto-aml-data-gen/simulate.py simulate.py
COPY auto-aml-data-gen/utils.py utils.py
RUN mkdir data

# Start with a bash shell
ENTRYPOINT ["python3", "main.py"]
