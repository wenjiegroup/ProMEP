import argparse
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    roc_curve)
from collections import defaultdict


def evaluate(labels, predictions, threshold=0.5):
    metrics = {}

    acc = accuracy_score(labels, predictions>=threshold)
    precision = precision_score(labels, predictions>=threshold)
    recall = recall_score(labels, predictions>=threshold)
    metrics["ACC"] = acc
    metrics["Precision"] =  precision
    metrics["Recall"] =  recall
    metrics["F1"] =  2*precision*recall/(precision+recall)

    aupr = average_precision_score(labels, predictions)
    metrics["AUPR"] =  aupr

    auroc = roc_auc_score(labels, predictions)
    metrics["AUROC"] =  auroc

    roc = roc_curve(labels, predictions)
    return metrics, roc

# def evaluate_roc_curve(labels, predictions):
    # auroc = roc_auc_score(labels, predictions)
    # fpr, tpr, thresholds = roc_curve(labels, predictions)
    # return auroc, fpr, tpr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default = 'testdata/pathogenicity_dataset/ddd.csv', type=str)
    parser.add_argument("--result", default = 'testdata/pathogenicity_dataset/pathogenicity_prediction.csv', type=str)
    parser.add_argument("--threshold", default = 6, type=float)
    parser.add_argument("--logits", type=str)
    parser.add_argument("--out", type=str)
    
    args = parser.parse_args()

    # read afm result and label
    afm_preds = []
    afm_labels = []
    lines = open(args.data, "r").readlines()
    for l in lines[1:]:
        items = l.strip().split(",")
        label = float(items[-1])
        pred = float(items[-2])

        afm_preds.append(pred)
        afm_labels.append(label)
    afm_preds, afm_labels = np.array(afm_preds), np.array(afm_labels)
    afm_metrics, afm_roc = evaluate(afm_labels, afm_preds)


    if args.logits is not None:
        # read logitsrange
        pro_lrange = {}
        for l in open(args.logits, "r").readlines():
            p, logits = l.strip().split("\t")
            logits = [float(s) for s in logits.strip().split(",")]
            p = p.split("_")[0]
            _min, _max = pro_lrange.get(p, (1000, -1000))
            _min = min(_min, min(logits))
            _max = max(_max, max(logits))
            pro_lrange[p] = (_min, _max)

        scores, am_scores, labels = [], [], []
        for l in open(args.result, "r").readlines():
            items = l.strip().split("\t")
            pro, var, score, mis, label = items
            score = (float(score) - pro_lrange[pro][0]) / (pro_lrange[pro][1] - pro_lrange[pro][0])
            scores.append(score)
            am_scores.append(float(mis))
            labels.append(float(label))
        scores, am_scores, labels = np.array(scores), np.array(am_scores), np.array(labels)
    else:
        # read result
        scores, am_scores, labels = [], [], []
        for l in open(args.result, "r").readlines():
            items = l.strip().split("\t")
            pro, var, score, mis, label = items
            scores.append(float(score))
            am_scores.append(float(mis))
            labels.append(float(label))
        scores, am_scores, labels = np.array(scores), np.array(am_scores), np.array(labels)

        # normalize score
        # scores = (scores-np.min(scores)) / (np.max(scores)-np.min(scores))
        # scores = scores - 9     # for data_s6
        # scores = scores - 6     # for data_s7
        scores = scores - args.threshold
        scores = np.exp(scores) / (1+np.exp(scores))
        # print(scores.shape, np.min(scores), np.max(scores))

    metrics, mep_roc = evaluate(labels=labels, predictions=scores)
    # am_metrics = evaluate(labels=labels, predictions=am_scores)

    print("ProMEP:")
    print(metrics)

    print("AlphaMissense")
    print(afm_metrics)

    # roc curve
    mep_fpr, mep_ftr, _ = mep_roc
    afm_fpr, afm_ftr, _ = afm_roc

    # print(len(mep_fpr), len(mep_ftr))
    # print(len(afm_fpr), len(afm_ftr))

    open('testdata/pathogenicity_dataset/ddd_roc_promep.tsv', "w").writelines([
        f"{fpr}\t{ftr}\n" for fpr, ftr in zip(mep_fpr, mep_ftr)
    ])

    open('testdata/pathogenicity_dataset/ddd_roc_afm.tsv', "w").writelines([
        f"{fpr}\t{ftr}\n" for fpr, ftr in zip(afm_fpr, afm_ftr)
    ])



if __name__ == "__main__":
    main()
