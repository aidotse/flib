import numpy as np
from sklearn.metrics import precision_recall_curve
from typing import Tuple, Union, List


def precision(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fp = confusion_matrix[:, 1]
    return np.divide(tp, tp + fp, where=(tp + fp) > 0, out=np.zeros_like(tp, dtype=float))


def recall(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fn = confusion_matrix[:, 2]
    return np.divide(tp, tp + fn, where=(tp + fn) > 0, out=np.ones_like(tp, dtype=float))


def average_precision_score(y_true: np.ndarray, y_pred: np.ndarray, recall_span: Tuple[int, int]=(0.0, 1.0)) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
    precisions = precisions[::-1]
    recalls = recalls[::-1]
    idxs = np.arange(np.where(recalls <= recall_span[0])[0][-1], np.where(recalls >= recall_span[1])[0][-1] + 1) # need "+ 1" due to np.arange 
    recall_diffs = np.diff(recalls[idxs])
    precision_areas = precisions[idxs[1:]] * recall_diffs
    return np.sum(precision_areas) / np.sum(recall_diffs)


def balanced_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    tp = confusion_matrix[:, 0]
    fp = confusion_matrix[:, 1]
    fn = confusion_matrix[:, 2]
    tn = confusion_matrix[:, 3]
    return 0.5*(tp/(tp+fn) + tn/(tn+fp))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, thresholds: Union[str, int, float, List[float]]) -> np.ndarray:
    if isinstance(thresholds, str) and thresholds == "dynamic":
        thresholds = np.unique(y_pred[:,1].round(decimals=4))
    if isinstance(thresholds, int):
        thresholds = np.linspace(0.0, 1.0, thresholds)
    if isinstance(thresholds, float):
        thresholds = np.array([thresholds])
    if isinstance(thresholds, list):
        thresholds = np.array(thresholds)
    y_true = y_true.astype(bool)
    cm = np.zeros((len(thresholds), 5), dtype=float)
    for i, threshold in enumerate(thresholds):
        cm[i, 0] = np.sum((y_pred[:,1] > threshold) & y_true)   # TP
        cm[i, 1] = np.sum((y_pred[:,1] <= threshold) & y_true)  # FP
        cm[i, 2] = np.sum((y_pred[:,1] > threshold) & ~y_true)  # FN
        cm[i, 3] = np.sum((y_pred[:,1] <= threshold) & ~y_true) # TN
        cm[i, 4] = threshold
    return cm
