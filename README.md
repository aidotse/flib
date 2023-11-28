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
* SWISH: Swedish Instant Payment System - a mobile payment system used in Sweden
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

## Specify parameters (with examples)

### conf.json

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
* **random_seed**

    The random seed is used to make the simulation reproducable.

* **simulation_name**

    The name of the simulation, used for naming the tmp and output folder.

* **total_steps**

    The total number of steps in the simulation. Each step is one day, but could be vewied as some other time unit.

* **min_amount, max_amount, mean_amount, std_amount, mean_amount_sar, std_amount_sar**

    The min and max amount of a transaction, and the mean and standard deviation of the truncated normal distribution used to sample the amount of a transaction. The distribution is truncated to zero and current blanace of the account. Mean and std are specifed for normal and SAR transactions.

* **prob_income, mean_income, std_income, prob_income_sar, mean_income_sar, std_income_sar**

    The probability for an account to recive income on a given step, and the mean and standard deviation of the truncated normal distribution used to sample the amount of the income.Mean and std are specifed for normal and SAR transactions.

* **mean_outcome, std_outcome, mean_outcome_sar, std_outcome_sar**

    The mean and standard deviation of the truncated normal distribution used to sample the amount of the outcome. Mean and std are specifed for normal and SAR transactions. The probability of an outcome in step $i$ is calculated from a sigmoid function: $p_i = 1 / (1 + e^{-x_i})$ where 
    $$x_i = \left( b_i - \frac{1}{N}\sum_{j=i-N}^{i} b_j  \right) \text{\huge/} \frac{1}{N}\sum_{j=i-N}^{i} b_j$$
    and $b_i$ is the balance of the account in step $i$ and $N$ is the number of steps in the past to consider.

* **mean_phone_change_frequency, std_phone_change_frequency, mean_phone_change_frequency_sar, std_phone_change_frequency_sar**

    The mean and standard deviation of the number of steps between a phone number change for an account. Mean and std are specifed for normal and SAR transactions. 

* **mean_bank_change_frequency, std_bank_change_frequency, mean_bank_change_frequency_sar, std_bank_change_frequency_sar**

    The mean and standard deviation of the number of steps between a bank change for an account. Mean and std are specifed for normal and SAR transactions.

* **margin_ratio**

    The margin a SAR account takes when passing money through it. 

### account.csv

The accounts.csv file contains the initial conditions for the accounts. It has the following columns:
* **count**: (int) The number of accounts to generate.
* **min_balance, max_balance**: (int) The minimum and maximum inital balance of the accounts. The inital balance is sampled from a uniform distribution.
* **country**: (string) The country of the accounts. 
* **business_type**: (string) The type of business of the accounts, OBS: currently only "I" is supported.
* **bank**: (string) The bank of the accounts. Can be left blank. 
Below is an example where 1000 accounts are generated in two banks with there different groups of inital balances.
``` 
count,min_balance,max_balance,country,business_type,bank
200,1000,10000,SWE,I,bank_a
200,10000,100000,SWE,I,bank_a
100,100000,200000,SWE,I,bank_a
200,1000,10000,SWE,I,bank_b
200,10000,100000,SWE,I,bank_b
100,100000,200000,SWE,I,bank_b
```

### normalModels.csv

normalModels.csv contains the normal transaction patterns of the accounts. It has the following columns:
* **count**: (int) The number of patterns to generate.
* **type**: (string) The type of the pattern. Can be single, fan_out, fan_in, forward, mutual or periodical. Se below for pattern definitions.
* **schedule_id**: (int) The id of the schedule to use for the pattern. Can be 0, 1, 2 or 3. Se below for schedule definitions.
* **min_accounts, max_accounts**: (int) The minimum and maximum number of accounts in the pattern. The simulator will find subsets of accounts where the pattern fits and sample from these. The number of subsets will depend on the min and max and on the structure of the network, defined in degree.csv. Some patterns has a fixed number of accounts, se pattern definition for more information.
* **min_period, max_period**: (int) The minimum and maximum period of the pattern. The period is the number of steps for a pattern to complet. The period is sampled from a uniform distribution.
* **bank_id**: (int) If specified, the patterns will only include accounts from that bank. If left blank, the patterns can include accounts from all banks.

Below is an example where 2000 different patterns are generated with varying number of accounts and periods. Some patterns are restricted to a specific bank, while others can include several banks.
```
count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id
100,single,0,2,2,1,84,bank_a
100,fan_in,1,6,8,21,21,bank_a
100,fan_out,2,6,10,7,14,bank_a
100,periodical,3,2,2,1,84,bank_a
100,single,0,2,2,1,84,bank_b
100,fan_in,1,6,8,21,21,bank_b
100,fan_out,2,6,10,7,14,bank_b
100,periodical,3,2,2,1,84,bank_b
300,forward,2,3,3,2,4,
300,mutual,2,2,2,1,10,
300,fan_out,2,12,16,28,56,
300,fan_in,2,10,20,56,84
```

### alertPatterns.csv
alertPatterns.csv contains the suspisious transaction patterns of the accounts. In contrast to normal models, the alert patterns will be place on top of the normal model transaction network. First the normal models will build a graph according to the degree.csv. Alert patterns will then be add into the graph, completly ignoring the degree.csv file. See (TODO: add section that clarifies this) for more information. alertPatterns.csv has the following columns:
* **count**: (int) The number of patterns to generate.
* **type**: (string) The type of the pattern. Can be fan_out, fan_in, cycle, random, bipartite, stack, gather_scatter or scatter_gather. Se below for pattern definitions.
* **schedule_id**: (int) The id of the schedule to use for the pattern. Can be 0, 1, 2 or 3. Se below for schedule definitions.
* **min_accounts, max_accounts**: (int) The minimum and maximum number of accounts in the pattern sampled from a uniform distribution. Some patterns has a minumum number of accounts, se pattern definition for more information. 
* **min_amount, max_amount**: (int) OBS: not used! 
* **min_period, max_period**: (int) The minimum and maximum period of the pattern. The period is the number of steps for a pattern to complet. The period is sampled from a uniform distribution.
* **bank_id**: (int) If specified, the patterns will only include accounts from that bank. If left blank, the patterns can include accounts from all banks.
* **is_sar**: (string) Can be true or false. If true, the pattern will be labeled as suspisious. If false, the pattern will be labeled as normal.
* **source_type**: (string) Can be CASH or TRANSFER. See (TODO: add section that clarifies this) for more information.

Below is an example with 8 alert patterns:
```
count,type,schedule_id,min_accounts,max_accounts,min_amount,max_amount,min_period,max_period,bank_id,is_sar,source_type
1,fan_out,2,10,20,100,1000,14,28,bank_a,True,CASH
1,fan_in,2,110,20,100,1000,14,28,bank_b,True,TRANSFER
1,cycle,2,10,20,100,1000,14,28,,True,TRANSFER
1,random,2,10,20,100,1000,14,28,,True,CASH
1,bipartite,2,10,20,100,1000,14,28,,True,TRANSFER
1,stack,2,10,20,100,1000,14,28,,True,TRANSFER
1,gather_scatter,2,10,20,100,1000,14,28,,True,CASH
1,scatter_gather,2,10,20,100,1000,14,28,,True,CASH
```

### degree.csv
The degree.csv file defines the structure of the transaction network. It has the following columns:
* **count**: (int) The number of nodes, aka accounts.
* **In-degree**: (int) The in-degree of the nodes.
* **Out-degree**: (int) The out-degree of the nodes.
The graph needs to be complet, i.e. the sum of the in-degree and out-degree of all nodes needs to be equal. Further, the total count needs to be equal to the number of accounts in the accounts.csv file. Below is an example for a graph with 1000 nodes. 
```
count,In-degree,Out-degree
512,10,10
256,20,20
128,30,30
64,40,40
32,50,50
16,60,60
8,70,70
4,80,80
2,90,90
```

### transactionType.csv

# Transaction Network Explorer

# Federated Learning

# TabDDPM
