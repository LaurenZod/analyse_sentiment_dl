# scripts/run_logreg.py
import os, re, time, argparse, subprocess, joblib
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
# Configuration simple via CLI
# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression on Sentiment140 and log to MLflow")
    ap.add_argument("--data", type=str, required=True,
        help="Chemin vers training.1600000.processed.noemoticon.csv")
    ap.add_argument("--experiment", type=str, default="baseline_tfidf_lr",
        help="Nom de l'expérience MLflow")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--subset_rows", type=int, default=None,
        help="Si défini, ne charge que ce nombre de lignes (pour aller vite).")
    return ap.parse_args()

# ------------------------------------------------------------------
# Prétraitement léger (cohérent avec ton notebook)
# ------------------------------------------------------------------
URL = re.compile(r'https?://\S+|www\.\S+')
USER = re.compile(r'@\w+')
HASH = re.compile(r'#(\w+)')
WS   = re.compile(r'\s+')

def transform_bow(t: str) -> str:
    t = URL.sub(' ', str(t))
    t = USER.sub(' ', t)
    t = HASH.sub(r'\1', t)   # garde le mot sans '#'
    t = WS.sub(' ', t).strip().lower()
    return t

# ------------------------------------------------------------------
# Utilitaires logging
# ------------------------------------------------------------------
def git_info():
    def _git(args):
        try:
            return subprocess.check_output(["git"]+args, text=True).strip()
        except Exception:
            return None
    return {
        "git.commit": _git(["rev-parse", "--short", "HEAD"]),
        "git.branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git.remote": _git(["config", "--get", "remote.origin.url"]),
    }

def plot_and_log_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["non_neg","neg"]); plt.yticks(ticks, ["non_neg","neg"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Pred"); plt.ylabel("True")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

def plot_and_log_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    mlflow.log_metric("auc", float(roc_auc))
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # 1) Initialiser MLflow (stockage local ./mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # 2) Charger Sentiment140
    read_kwargs = dict(
        header=None,
        names=["target","ids","date","flag","user","text"],
        sep=",", encoding="ISO-8859-1", quotechar='"', engine="python"
    )
    if args.subset_rows:
        df = pd.read_csv(args.data, nrows=args.subset_rows, **read_kwargs)
        # ATTENTION : le CSV est souvent trié par polarité. Pour éviter la mono-classe :
        # On mélange rapidement :
        df = df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    else:
        df = pd.read_csv(args.data, **read_kwargs)

    # Label binaire cohérent avec ton entraînement précédent :
    # 1 = négatif (target=0), 0 = non-négatif (target=4)
    df["label"] = df["target"].map({0:1, 4:0}).astype(int)
    df = df[["text","label"]].dropna()

    X = df["text"].astype(str).apply(transform_bow)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 3) Entraînement + logging MLflow
    run_name = f"logreg_tfidf_C{args.C}_ng12"
    with mlflow.start_run(run_name=run_name):
        # Tags Git
        for k, v in git_info().items():
            if v: mlflow.set_tag(k, v)

        # Log params
        mlflow.log_params({
            "vectorizer": "tfidf",
            "ngram_range": f"(1,{args.ngram_max})",
            "max_features": args.max_features,
            "min_df": args.min_df,
            "clf": "LogisticRegression",
            "C": args.C,
            "solver": "liblinear",
            "class_weight": "balanced",
            "test_size": args.test_size,
            "subset_rows": args.subset_rows,
        })

        # Vectorizer + modèle
        vec = TfidfVectorizer(ngram_range=(1, args.ngram_max),
                              max_features=args.max_features, min_df=args.min_df)
        t0 = time.perf_counter()
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=args.C,
                                 class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y_train)
        dur = time.perf_counter() - t0
        mlflow.log_metric("duration_sec", float(dur))

        # Metrics
        y_pred = clf.predict(Xte)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("accuracy", float(acc))

        # Courbe ROC (probabilité classe 1 = négatif)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(Xte)[:, 1]
            plot_and_log_roc(y_test, y_score)

        # Confusion + rapport texte
        plot_and_log_confusion(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Sauvegarde du pack (vectorizer + modèle) pour réutiliser l'inférence
        os.makedirs("models/baseline", exist_ok=True)
        out = "models/baseline/tfidf_logreg.joblib"
        joblib.dump({"vectorizer": vec, "model": clf}, out)
        mlflow.log_artifact(out)

        # Sauvegarde aussi au format MLflow (modèle seul)
        mlflow.sklearn.log_model(clf, artifact_path="sk_model")

        print(f"✅ F1_macro: {f1:.4f} | accuracy: {acc:.4f} | duration_sec: {dur:.2f}")

if __name__ == "__main__":
    main()
