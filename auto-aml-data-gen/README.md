sudo code
1. define SAR/normal ratio
2. define bounds for params
3. define constraints, all features should have similar feature importance
4. define recall
5. set target FPR
3. set all distributions equal, simulate data, train with hyperparamter tuning, and evaluate FPR at given recall -> if needed use FPR to normalize 
4. loop: 
    suggest AMLSim parameters that satisfies the target SAR/normal ratio
    create aml data
    train (with hyperparameter tuning) statistical models
    check constraints
    evaluate FPR for given recall
    end if FPR are satisfied


TODO:
- [x] function to generate config files
- [x] function to start simulation
- [ ] function to train and hyperparameter tune models
- [ ] function to check constrains
- [ ] function to evaluate FPR
- [ ] function to suggest new parameters