import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    fbeta_score, 
    precision_recall_curve,
    auc,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# useful function to load the dataset
def load_dataset(dataset_path, target_name):
    # load the dataset
    data = np.genfromtxt(dataset_path, delimiter=',', skip_header=1)

    # load the feature names
    with open(dataset_path, 'r') as f:
        feature_names = f.readline().strip().split(',')

    # removing the target from the design matrix X
    target_idx = feature_names.index(target_name)
    feature_names.remove(target_name)

    y = data[:, target_idx]
    X = np.delete(data, target_idx, axis=1)

    return X, y

def feat_names(dataset_path, target_name):
    feat_names = None
    with open(dataset_path, "r") as f:
        feature_names = f.readline().strip().split(",")

    feature_names.remove(target_name)
    return feature_names


# useful function to compute metrics
def compute_metrics(true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    
    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = fbeta_score(true, preds, beta=1, zero_division=0)
    f2 = fbeta_score(true, preds, beta=2, zero_division=0)
    precisions, recalls, _ = precision_recall_curve(true, probs)
    auprc = auc(recalls, precisions)
    
    table = (
        f"{'='*25}\n"
        f"{'Metric':<10} | \n"
        f"{'='*25}\n"
        f"{'Accuracy':<10} | {acc:<10.4f}\n"
        f"{'Precision':<10} | {prec:<10.4f}\n"
        f"{'Recall':<10} | {rec:<10.4f}\n"
        f"{'F1-Score':<10} | {f1:<10.4f}\n"
        f"{'F2-Score':<10} | {f2:<10.4f}\n"
        f"{'AUPRC':<10} | {auprc:<10.4f}\n"
        f"{'='*25}\n"
    )
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # confusion matrix
    cm = confusion_matrix(true, preds)
    tn, fp, fn, tp = cm.ravel()
    labels = np.array([
        [f"TN\n{tn/(tn+fp)*100:.2f}", f"FP\n{fp/(tn+fp)*100:.2f}"],
        [f"FN\n{fn/(tp+fn)*100:.2f}", f"TP\n{tp/(tp+fn)*100:.2f}"]
    ])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"], ax=ax1)
    ax1.set_title(f"Confusion Matrix")
    
    # precision-recall curve
    ax2.plot(recalls, precisions, label=f'PR Curve (AUPRC = {auprc:.4f})')
    
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    metriche_dict = {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, 
        "f2": f2, "auprc": auprc, "confusion_matrix": cm
    }
    
    return table, fig, metriche_dict