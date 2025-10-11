import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import mlflow

def log_confusion(y_true, y_pred, labels=("neg","pos")):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix"); plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Pred"); plt.ylabel("True")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

def log_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    mlflow.log_metric("auc", float(roc_auc))
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)