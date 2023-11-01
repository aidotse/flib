# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

WORKDIR /amlsim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.7 && \
    apt-get install -y python3-pip && \
    apt-get install -y wget && \
    apt-get install -y software-properties-common && \
    apt-get install -y apt-transport-https && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y maven && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY jars jars

RUN mvn install:install-file \
    -Dfile=jars/mason.20.jar \
    -DgroupId=mason \
    -DartifactId=mason \
    -Dversion=20 \
    -Dpackaging=jar \
    -DgeneratePom=true

COPY scripts scripts

COPY src src
COPY janusgraph.properties janusgraph.properties
COPY index.html index.html
COPY pom.xml pom.xml
COPY target target

ENTRYPOINT ["sh", "scripts/run.sh"]