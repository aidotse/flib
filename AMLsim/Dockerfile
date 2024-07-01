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
    apt-get install -y jq && \
    apt-get install -y nano && \
    #apt-get install -y maven && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# Set environment variables for Maven
ENV MAVEN_VERSION 3.9.6
ENV MAVEN_HOME /usr/share/maven

# Change to tmp folder
WORKDIR /tmp

# Download and install Maven
RUN wget https://downloads.apache.org/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz -O - | tar xzf - -C /usr/share \
    && ln -s /usr/share/apache-maven-${MAVEN_VERSION} /usr/share/maven \
    && ln -s /usr/share/maven/bin/mvn /usr/bin/mvn

# Cleanup
RUN rm -rf /tmp/*

WORKDIR /amlsim

#RUN apt update && apt upgrade -y
#
#RUN apt install wget -y
#
#RUN apt install default-jdk -y 
#RUN wget https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz 
#RUN tar xzf apache-maven-3.9.6-bin.tar.gz -C /opt/ 
#
#WORKDIR /opt
#
#RUN ln -s apache-maven-3.9.6 maven 
#RUN touch /etc/profile.d/maven.sh
#RUN echo "export M2_HOME=/opt/maven\nexport PATH=\${M2_HOME}/bin:\${PATH}" | tee /etc/profile.d/maven.sh 

#RUN ["source", "/etc/profile.d/maven.sh"]
#CMD ["sh", "/etc/profile.d/maven.sh"]

#WORKDIR /amlsim

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

RUN mvn clean package -DskipTests

RUN sh scripts/run.sh

ENTRYPOINT ["bash", "scripts/run.sh"]
