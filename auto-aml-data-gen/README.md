## sudo code
1. define SAR/normal ratio
2. define bounds for params
3. define operating recall
4. set target FPR
5. set all distributions equal, simulate data, train with hyperparamter tuning, and evaluate FPR at given recall -> if needed use FPR to normalize 
6. loop for n trials: 
    - suggest new AMLSim temporal parameters via multivariate optimization on FPR and feature importance
    - create aml data
    - preprocess data
    - train (with hyperparameter tuning) statistical model, e.g. random forest
    - evaluate FPR and feature importance

TODO:
- [x] function to generate config files
- [x] function to start simulation
- [x] function to preprocess data
- [x] function to train and hyperparameter tune models
- [x] function to check constraint on feature importance
- [x] function to evaluate FPR
- [x] function to suggest new parameters, use optuna! many loss functions, one for each parameter?