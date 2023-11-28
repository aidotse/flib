# FLIB: Federated Learning in Banking

** OBS: under construction, most code dont run right away... **

This is the repsitory for all the code in the project.

## Currently containing

# AMLsim
AMLsim is a simulator for generating transaction networks used in anti-money laundering research. It is based on the simulator by IBM (TODO: add link) and is extended to utilize distributions and model behavioural features. This version is designed to generate SWISH data of personal accounts. It can simulate income and outcome for accounts, as well as known transactions patterns of normal and suspisious behaviour. In short, it has two parts: a python part for generating the transaction network and a java part for simulating the behaviour of the agents. The simulation is controlled by 6 parameter files:
* 1 json file, which defines behviours of accounts and some paths varibles used during the simulation. 
* 5 csv files, which defines some inital condtions and together defines the structure of the transaction network.

The output of the simulation is a csv file with all the transactions.

## Accronyms and definitions
* SAR: Suspicious Activity Report - accounts or transactions that are labeled as suspisious by the bank
* SWISH: Swedish Instant Payment System - a payment system used in Sweden
* AML: Anti-Money Laundering - the process of detecting and preventing money laundering
* Transaction: A SWISH transaction between two accounts
* Income: A transaction from a source to and an account (not a SWISH transaction)
* Outcome: A transaction from an account to a sink (not a SWISH transaction)

## Dependencies

### Alternative 1: Docker

1. pull image from thecoldice/amlsim:latest on dockerhub

### Alternative 2: Manual

Dependencies: python3.7, java, maven

1. clone repo
2. move into AMlsim folder
3. install python dependencies: `pip install -r requirements.txt` or `conda env create -f AMLamlsim.yml`
4. install java dependencies: `mvn install:install-file -Dfile=jars/mason.20.jar -DgroupId=mason -DartifactId=mason -Dversion=20 -Dpackaging=jar -DgeneratePom=true`

## Setup

1. Create a folder for the outputs: `mkdir outputs`
2. (Only for manual) Create a temporary folder for storing pyhton output: `mkdir tmp`
3. Create a folder for the simulation paramters: `mkdir paramFiles`
4. In paramFiles create a folder for a new simulation, e.g. `mkdir paramFiles/simulation1`
5. In the simulation folder, create these files: conf.json, accounts.csv, normalModels.csv, alertPatterns.csv, degree.csv and transactionTypes.csv
6. Specify the parameters in the files (see below)

### Specify parameters (with examples)

#### conf.json
The conf.json file contains parameters for the generel behaviour of the accounts and paths to the other files, the paths are relative to the conf.json file. A example looks like this:
```
{
  "general": {
    "random_seed": 0,
    "simulation_name": "simulation1",
    "total_steps": 86
  },
  "default": {
    "min_amount": 1,
    "max_amount": 150000,
    "mean_amount": 637,
    "std_amount": 1000,
    "mean_amount_sar": 2000,
    "std_amount_sar": 1000,
    "prob_income": 0.05,
    "mean_income": 500.0,
    "std_income": 1000.0,
    "prob_income_sar": 0.05,
    "mean_income_sar": 500.0,
    "std_income_sar": 1000.0,
    "mean_outcome": 200.0,
    "std_outcome": 500.0,
    "mean_outcome_sar": 200.0,
    "std_outcome_sar": 500.0,
    "mean_phone_change_frequency": 1460,
    "std_phone_change_frequency": 365,
    "mean_phone_change_frequency_sar": 365,
    "std_phone_change_frequency_sar": 182,
    "mean_bank_change_frequency": 1460,
    "std_bank_change_frequency": 365,
    "mean_bank_change_frequency_sar": 365,
    "std_bank_change_frequency_sar": 182,
    "margin_ratio": 0.1
  },
  "input": {
    "directory": "paramFiles/simulation1",
    "schema": "schema.json",
    "accounts": "accounts.csv",
    "alert_patterns": "alertPatterns.csv",
    "normal_models": "normalModels.csv",
    "degree": "degree.csv",
    "transaction_type": "transactionType.csv",
    "is_aggregated_accounts": true
  },
  "temporal": {
    "directory": "tmp",
    "transactions": "transactions.csv",
    "accounts": "accounts.csv",
    "alert_members": "alert_members.csv",
    "normal_models": "normal_models.csv"
  },
  "output": {
    "directory": "outputs",
    "transaction_log": "tx_log.csv"
  },
  "graph_generator": {
    "degree_threshold": 1
  },
  "simulator": {
    "transaction_limit": 100000,
    "transaction_interval": 7,
    "sar_interval": 7
  }
}
```
##### random_seed
The random seed is used to make the simulation reproducable.

##### simulation_name
The name of the simulation, used for naming the tmp and output folder.

##### total_steps
The total number of steps in the simulation. Each step is one day, but could be vewied as some other time unit.

##### min_amount, max_amount, mean_amount, std_amount, mean_amount_sar, std_amount_sar
The min and max amount of a transaction, and the mean and standard deviation of the truncated normal distribution used to sample the amount of a transaction. The distribution is truncated to zero and current blanace of the account. Mean and std are specifed for normal and SAR transactions.

##### prob_income, mean_income, std_income, prob_income_sar, mean_income_sar, std_income_sar
The probability for an account to recive income on a given step, and the mean and standard deviation of the truncated normal distribution used to sample the amount of the income.Mean and std are specifed for normal and SAR transactions.

##### mean_outcome, std_outcome, mean_outcome_sar, std_outcome_sar
The mean and standard deviation of the truncated normal distribution used to sample the amount of the outcome. Mean and std are specifed for normal and SAR transactions. The probability of an outcome calculated form a sigmoid function: 
$$p_i = sigmoid( 1/N \sum_{j=i-N}^{i} balance_j )$$

# Transaction Network Explorer

# Federated Learning

# TabDDPM
