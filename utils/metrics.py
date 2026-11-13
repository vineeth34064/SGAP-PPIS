import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute full suite of PPIS evaluation metrics:
    - ACC, Precision, Recall, F1, MCC, AUROC, AUPRC
    - TP, TN, FP, FN
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_prob = np.asarray(y_prob).astype(float).flatten()

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    try:
        if len(np.unique(y_true)) > 1:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            auprc = auc(rec_arr, prec_arr)
        else:
            auprc = 0.0
    except Exception:
        auprc = 0.0

    return {
        'ACC': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'MCC': float(mcc),
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Threshold': float(threshold)
    }

def find_optimal_threshold(y_true, y_prob, metric='MCC', step=0.01):
    """
    Find optimal classification threshold on validation set (Gap 6 rectification).
    Due to class imbalance and focal loss weighting, standard 0.5 threshold is biased.
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_prob = np.asarray(y_prob).astype(float).flatten()

    best_thresh = 0.5
    best_score = -float('inf')

    thresholds = np.arange(0.05, 0.95, step)
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        if metric == 'MCC':
            score = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0
        elif metric == 'F1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_thresh = th

    return float(best_thresh), float(best_score)
