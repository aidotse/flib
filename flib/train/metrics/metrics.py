import numpy as np
from typing import Dict, Tuple


def calculate_precision(tp:int, fp:int) -> float:
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    return precision

def calculate_recall(tp:int, fn:int) -> float:
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    return recall

def calculate_average_precision(tpfptnfn:Dict[int, Dict[str, int]], recall_span:Tuple[int, int]=(0.0, 1.0)) -> float:
    avg_precision = 0.0
    n = 0
    for threshold in tpfptnfn:
        rec = calculate_recall(tpfptnfn[threshold]['tp'], tpfptnfn[threshold]['fn'])
        if rec >= recall_span[0] and rec <= recall_span[1]:
            pre = calculate_precision(tpfptnfn[threshold]['tp'], tpfptnfn[threshold]['fp'])
            avg_precision += pre
            n += 1
    avg_precision = avg_precision / n
    return avg_precision