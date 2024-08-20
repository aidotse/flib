from scipy import stats
import numpy as np
import os
import json

def powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0, seed=0):
    
    degrees = stats.pareto.rvs(gamma, loc=loc, scale=scale, size=n, random_state=seed).round()
    
    if degrees.sum() % 2 == 1:
        degrees[np.random.randint(n)] += 1
    split = np.random.rand(n)
    
    in_degrees = (degrees*split).round()
    out_degrees = (degrees*(1-split)).round()
    
    iters = 0
    while in_degrees.sum() != out_degrees.sum() and iters < 10000:
        if in_degrees.sum() > out_degrees.sum():
            idx = np.random.choice(np.where(in_degrees > 1.0)[0])
            in_degrees[idx] -= 1
            out_degrees[np.random.randint(n)] += 1
        else:
            idx = np.random.choice(np.where(out_degrees > 1.0)[0])
            in_degrees[np.random.randint(n)] += 1
            out_degrees[idx] -= 1
        iters += 1
    if in_degrees.sum() > out_degrees.sum():
        diff = in_degrees.sum() - out_degrees.sum()
        assert diff % 2 == 0
        in_degrees[np.argmax(in_degrees)] -= diff / 2
        out_degrees[np.argmax(out_degrees)] += diff / 2
    elif in_degrees.sum() < out_degrees.sum():
        diff = out_degrees.sum() - in_degrees.sum()
        assert diff % 2 == 0
        in_degrees[np.argmax(in_degrees)] += diff / 2
        out_degrees[np.argmax(out_degrees)] -= diff / 2
    
    degrees = np.column_stack((in_degrees,out_degrees))
    
    values, counts = np.unique(degrees, return_counts=True, axis=0)
    
    return values, counts

def read_bounds(folder:str):
    
    conf_file = os.path.join(folder, 'conf.json')
    with open(conf_file, 'r') as f:
        conf_params = json.load(f)
    
    conf_bounds = {}
    for k1 in conf_params:
        conf_bounds[k1] = {}
        for k2 in conf_params[k1]:
            if type(conf_params[k1][k2]) == list:
                conf_bounds[k1][k2] = {'lb': conf_params[k1][k2][0], 'ub': conf_params[k1][k2][1], }
            else:
                conf_bounds[k1][k2] = {'lb': conf_params[k1][k2], 'ub': conf_params[k1][k2]}

    # TODO: read bounds from csv files as well
    
    return conf_bounds


read_bounds('param_files/test')