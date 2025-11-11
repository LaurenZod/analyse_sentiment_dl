# scripts/run_logreg.py
import matplotlib
matplotlib.use("Agg")
import os, re, time, argparse, subprocess, joblib, random
import numpy as np
import pandas as pd
from packaging import version
from mlflow.models.signature import infer_signature
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt
from scripts.utils_text import transform_bow

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
# CLI
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
# Utils logging
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
    plt.xticks(ticks, ["neg","pos"]); plt.yticks(ticks, ["neg","pos"])
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

    # Seeds (reproductibilité)
    np.random.seed(args.random_state)
    random.seed(args.random_state)

    # MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # Lecture CSV (Sentiment140)
    read_kwargs = dict(
        header=None,
        names=["target","ids","date","flag","user","text"],
        sep=",", encoding="ISO-8859-1", quotechar='"', engine="python"
    )
    if args.subset_rows:
        df_full = pd.read_csv(args.data, **read_kwargs)
        df_full["label"] = df_full["target"].map({0:0, 4:1})
        df_full.dropna(subset=["label"], inplace=True)
        df_full["label"] = df_full["label"].astype(int)

        n = args.subset_rows // 2
        # Sécurise si une classe a moins que n (label: 0 = négatif, 1 = positif)
        n_neg = min(n, (df_full["label"] == 0).sum())
        n_pos = min(n, (df_full["label"] == 1).sum())
        df_neg = df_full[df_full["label"] == 0].sample(n=n_neg, random_state=args.random_state)
        df_pos = df_full[df_full["label"] == 1].sample(n=n_pos, random_state=args.random_state)
        df = pd.concat([df_neg, df_pos]).sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    else:
        df = pd.read_csv(args.data, **read_kwargs)
        df["label"] = df["target"].map({0:0, 4:1})
        df.dropna(subset=["label"], inplace = True)
        df = df[["text","label"]]
        df["label"] = df["label"].astype(int)

    # Prétraitement léger
    X = df["text"].astype(str).apply(transform_bow)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Entraînement + logging
    run_name = f"logreg_tfidf_C{args.C}_ng1{args.ngram_max}"
    with mlflow.start_run(run_name=run_name):
        for k, v in git_info().items():
            if v: mlflow.set_tag(k, v)

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
            "seed": args.random_state
        })

        vec = TfidfVectorizer(
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            min_df=args.min_df
        )

        t0 = time.perf_counter()
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)

        clf = LogisticRegression(
            max_iter=1000, C=args.C, class_weight="balanced", solver="liblinear"
        )
        clf.fit(Xtr, y_train)
        dur = time.perf_counter() - t0
        mlflow.log_metric("duration_sec", float(dur))

        y_pred = clf.predict(Xte)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("accuracy", float(acc))

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(Xte)[:, 1]
            plot_and_log_roc(y_test, y_score)

        plot_and_log_confusion(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)
        mlflow.log_text(report, "classification_report.txt")

        os.makedirs("models/baseline", exist_ok=True)
        out = "models/baseline/tfidf_logreg.joblib"
        joblib.dump({"vectorizer": vec, "model": clf}, out)
        mlflow.log_artifact(out)

        # Sauvegarde aussi au format MLflow (modèle seul)

        input_example = Xtr[:3].toarray()
        signature = infer_signature(Xtr[:10].toarray(),
                                    clf.predict_proba(Xtr[:10])[:, 1])

        if version.parse(mlflow.__version__) >= version.parse("3.4.0"):
            mlflow.sklearn.log_model(
                clf,
                name="sk_model",
                input_example=input_example,
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(
                clf,
                artifact_path="sk_model",
                input_example=input_example,
                signature=signature,
            )

        print(f"✅ F1_macro: {f1:.4f} | accuracy: {acc:.4f} | duration_sec: {dur:.2f}")

if __name__ == "__main__":
    main()