"""
Generate scale-free graph and output degree-distribution CSV file
"""

import numpy as np
import networkx as nx
from collections import Counter
import csv
import sys
import json
import os
from scipy import stats

def kronecker_generator(scale, edge_factor):
    """Kronecker graph generator
  Ported from octave code in https://graph500.org/?page_id=12#alg:generator
  """
    N = 2 ** scale  # Number of vertices
    M = N * edge_factor  # Number of edges
    A, B, C = (0.57, 0.19, 0.19)  # Initiator probabilities
    ijw = np.ones((3, M))  # Index arrays

    ab = A + B
    c_norm = C / (1 - (A + B))
    a_norm = A / (A + B)

    for ib in range(scale):
        ii_bit = (np.random.rand(1, M) > ab).astype(int)
        ac = c_norm * ii_bit + a_norm * (1 - ii_bit)
        jj_bit = (np.random.rand(1, M) > ac).astype(int)
        ijw[:2, :] = ijw[:2, :] + 2 ** ib * np.vstack((ii_bit, jj_bit))

    ijw[2, :] = np.random.rand(1, M)
    ijw[:2, :] = ijw[:2, :] - 1
    q = range(M)
    np.random.shuffle(q)
    ijw = ijw[:, q]
    edges = ijw[:2, :].astype(int).T
    _g = nx.DiGraph()
    _g.add_edges_from(edges)
    return _g


def kronecker_generator_general(_n, _m):
    # TODO: Accept general number of nodes and edges
    A, B, C = (0.57, 0.19, 0.19)  # Initiator probabilities
    ijw = np.ones((3, _m))  # Index arrays
    ab = A + B
    c_norm = C / (1 - (A + B))
    a_norm = A / (A + B)
    tmp = _n
    while tmp > 0:
        tmp //= 2
        ii_bit = (np.random.rand(1, _m) > ab).astype(int)
        ac = c_norm * ii_bit + a_norm * (1 - ii_bit)
        jj_bit = (np.random.rand(1, _m) > ac).astype(int)
        ijw[:2, :] = ijw[:2, :] + tmp * np.vstack((ii_bit, jj_bit))
    ijw[2, :] = np.random.rand(1, _m)
    ijw[:2, :] = ijw[:2, :] - 1
    q = range(_m)
    np.random.shuffle(q)
    ijw = ijw[:, q]
    edges = ijw[:2, :].astype(int).T
    _g = nx.DiGraph()
    _g.add_edges_from(edges)
    return _g


def powerlaw_cluster_generator(_n, _edge_factor):
    edges = nx.barabasi_albert_graph(_n, _edge_factor, seed=0).edges()  # Undirected edges

    # Swap the direction of half edges to diffuse degree
    di_edges = [(edges[i][0], edges[i][1]) if i % 2 == 0 else (edges[i][1], edges[i][0]) for i in range(len(edges))]
    _g = nx.DiGraph()
    _g.add_edges_from(di_edges)  # Create a directed graph
    return _g


def get_n(conf_file):
    # open conf.json and get accounts file
    with open(conf_file, "r") as rf:
        conf = json.load(rf)
        directory = conf["input"]["directory"]
        accounts_file = conf["input"]["accounts"]
    # build accounts file path
    accounts_path = directory + '/' + accounts_file
    # open accounts file and get number of accounts
    with open(accounts_path, "r") as rf:
        # skip header
        next(rf)
        # sum first int on each line
        n = sum([int(line.split(',')[0]) for line in rf])
    return n

def get_edge_factor(conf_file):
    with open(conf_file, "r") as rf:
        conf = json.load(rf)
        edge_factor = conf["default"]["edge_factor"]
    return edge_factor

def get_scale_free_params(conf_file):
    with open(conf_file, "r") as rf:
        conf = json.load(rf)
        scale_free_params = conf["scale-free"]
        if "gamma" in scale_free_params:
            gamma = scale_free_params["gamma"]
        else:
            gamma = 2.0
        if "loc" in scale_free_params:
            loc = scale_free_params["loc"]
        else:
            loc = 1.0
        if "scale" in scale_free_params:
            scale = scale_free_params["scale"]
        else:
            scale = 1.0
    return gamma, loc, scale

def plot_powerlaw_degree_distrubution(n, gamma=2, edge_factor=20, scale=1.0, min_degree=1, values=None, counts=None, alp=None, bet=None, gam=None):
    
    import matplotlib.pyplot as plt
    
    def func(x, scale, gamma):
        return scale * np.power(x, -gamma)
    
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, 1000, 1000)
    
    plt.plot(x, func(x, scale, gamma), label=f'reference\n  gamma={gamma:.2f}\n  scale={scale:.2f}', color='C0')
    
    if values is not None and counts is not None:
        probs = counts / n
        log_values = np.log(values)
        log_probs = np.log(probs)
        coeffs = np.polyfit(log_values, log_probs, 1)
        gamma, scale = coeffs
        print(f'pareto sampling: gamma={gamma}, scale={np.exp(scale)}')
        plt.plot(x, func(x, np.exp(scale), -gamma), label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}, min_deg={min_degree}', color='C1')
        plt.scatter(values, probs, label='original', color='C1')
    else:
        degrees = (min_degree + scale * np.random.pareto(gamma, n)).round()
        pareto_values, pareto_counts = np.unique(degrees, return_counts=True)
        pareto_probs = pareto_counts / n
        pareto_log_values = np.log(pareto_values)
        pareto_log_probs = np.log(pareto_probs)
        pareto_coeffs = np.polyfit(pareto_log_values, pareto_log_probs, 1)
        pareto_gamma, pareto_scale = pareto_coeffs
        print(f'pareto sampling: gamma={pareto_gamma}, scale={np.exp(pareto_scale)}')
        plt.plot(x, func(x, np.exp(pareto_scale), -pareto_gamma), label=f'pareto sampling fit\n  gamma={-pareto_gamma:.2f}\n  scale={np.exp(pareto_scale):.2f}\n  min_deg={min_degree}', color='C1')
        plt.scatter(pareto_values, pareto_probs, label='pareto sampling', color='C1')
    
    if edge_factor is not None:
        g = nx.barabasi_albert_graph(n, edge_factor)
        degrees = np.array(list(dict(g.degree()).values()))
        baralb_values, baralb_counts = np.unique(degrees, return_counts=True)
        baralb_probs = baralb_counts / n
        baralb_log_values = np.log(baralb_values)
        baralb_log_probs = np.log(baralb_probs)
        baralb_coeffs = np.polyfit(baralb_log_values, baralb_log_probs, 1)
        baralb_gamma, baralb_scale = baralb_coeffs
        print(f'barabasi-albert: gamma={baralb_coeffs[0]}, scale={np.exp(baralb_coeffs[1])}')
        plt.plot(x, func(x, np.exp(baralb_scale), -baralb_gamma), label=f'barabasi-albert fit\n  gamma={-baralb_gamma:.2f}\n  scale={np.exp(baralb_scale):.2f}', color='C2')
        plt.scatter(baralb_values, baralb_probs, label='barabasi-albert', color='C2')
    
    if alp is not None and bet is not None and gam is not None:
        g = nx.scale_free_graph(n, alpha=alp, beta=bet, gamma=gam)
        degrees = np.array(list(dict(g.degree()).values()))
        sf_values, sf_counts = np.unique(degrees, return_counts=True)
        sf_probs = sf_counts / n
        sf_log_values = np.log(sf_values)
        sf_log_probs = np.log(sf_probs)
        sf_coeffs = np.polyfit(sf_log_values, sf_log_probs, 1)
        sf_gamma, sf_scale = sf_coeffs
        print(f'scale-free: gamma={sf_coeffs[0]}, scale={np.exp(sf_coeffs[1])}')
        plt.plot(x, func(x, np.exp(sf_scale), -sf_gamma), label=f'scale-free fit\n  gamma={-sf_gamma:.2f}\n  scale={np.exp(sf_scale):.2f}', color='C3')
        plt.scatter(sf_values, sf_probs, label='scale-free', color='C3')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('log(degree)')
    plt.ylabel('log(probability)')
    plt.legend()
    plt.grid()
    
    # save plot
    plt.savefig('degree_distributions.png')

def powerlaw_degree_distrubution(n, gamma=2.0, loc=1.0, scale=1.0):
    
    degrees = stats.pareto.rvs(gamma, loc=loc, scale=scale, size=n).round()
    
    # if degree sum is odd, add one to a random degree
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
        in_degrees[np.argmax(in_degrees)] -= diff
        out_degrees[np.argmax(out_degrees)] += diff
    elif in_degrees.sum() < out_degrees.sum():
        diff = out_degrees.sum() - in_degrees.sum()
        in_degrees[np.argmax(in_degrees)] += diff
        out_degrees[np.argmax(out_degrees)] -= diff
    
    degrees = np.column_stack((in_degrees,out_degrees))
    
    values, counts = np.unique(degrees, return_counts=True, axis=0)
    
    return values, counts
    

if __name__ == "__main__":
    
    argv = sys.argv
    
    if len(argv) == 2:
        conf_file = argv[1]
        # get number of accounts from conf.json
        n = get_n(conf_file)
        # get egde factor from conf.json
        gamma, loc, scale = get_scale_free_params(conf_file)
        # get directory from conf.json
        with open(conf_file, "r") as rf:
            conf = json.load(rf)
            directory = conf["input"]["directory"]
            deg_file = conf["input"]["degree"]
        # build degree file path
        deg_file_path = os.path.join(directory, deg_file)
    elif len(argv) == 1:
        deg_file_path = "degree.csv"
        # parameters for pareto sampling
        n = 10000
        gamma = 2.0
        loc = 1.0
        scale = 1.0
        
    values, counts = powerlaw_degree_distrubution(n, gamma, loc, scale)
    
    with open(deg_file_path, "w") as wf:
        writer = csv.writer(wf)
        writer.writerow(["Count", "In-degree", "Out-degree"])
        for value, count in zip(values, counts):
            writer.writerow([count, int(value[0]), int(value[1])])
