sudo code
1. define SAR/normal ratio
2. define bounds for params
3. define constraints, all features should have similar feature importance
4. define recall
5. set target FPR
3. set all distributions equal, simulate data, train and evaluate recall and FPR
    if needed use recall and FPR to normalize 
4. loop: 
    suggest config files that satisfies the target SAR/normal ratio
    create aml data
    train statistical models
    end if recall and FPR are satisfied