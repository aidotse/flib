# FLIB: Federated Learning in Banking

** OBS: under construction, most code dont run right away... **

This is the repsitory for all the code in the project.

## Currently containing

# AMLsim
AMLsim is a simulator for generating transaction networks used in anti-money laundering research. It is based on the simulator by IBM (TODO: add link) and is extended to utilize distributions and model behavioural features. In short, it has two parts: a python part for generating the transaction network and a java part for simulating the behaviour of the agents. The simulation is controlled by 6 parameter files. 
* A json file, which defines behviours of accounts and some paths varibles used during the simulation. 
* 5 csv files, which defines some inital condtions and together defines the structure of the transaction network.

## Dependencies

### Alternative 1: Docker

1. pull image from thecoldice/amlsim:latest on dockerhub

### Alternative 2: Manual

Dependencies: python3.7, java, maven

1. clone repo
2. move into AMlsim folder
3. install python dependencies: `pip install -r requirements.txt` or `conda env create -f AMLamlsim.yml`
4. install java dependencies: `mvn install:install-file -Dfile=jars/mason.20.jar -DgroupId=mason -DartifactId=mason -Dversion=20 -Dpackaging=jar -DgeneratePom=true`
    `
## Setup

1. Create a folder for the outputs: `mkdir outputs`
2. (Only for manual) Create a temporary folder for storing pyhton output: `mkdir tmp`
2. Create a folder for the simulation paramters: `mkdir paramFiles`
2. In paramFiles create a folder for a new simulation, e.g. `mkdir paramFiles/simulation1`
3. In the simulation folder, create these files: conf.json, accounts.csv, normalModels.csv, alertPatterns.csv, degree.csv and transactionTypes.csv

Transaction Network Explorer

Federated Learning

TabDDPM
