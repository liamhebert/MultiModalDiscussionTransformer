from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)

import numpy as np, wandb, hashlib


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    # compute roc_auc score
    x = np.exp(pred.predictions)
    # the part in parentheses is softmax
    predict_probas = (x / x.sum(axis=-1, keepdims=True))[:, 1]
    auc = roc_auc_score(labels, predict_probas)

    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc}


def wandb_init(exp_dict):
    wandb.init(project="hatespeech_detection", config=exp_dict)


def hash_str(string):
    return hashlib.md5(string.encode()).hexdigest()
