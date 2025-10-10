# scripts/run_use_mlp.py
import os, time, argparse, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="USE + MLP on Sentiment140 with MLflow logging")
    ap.add_argument("--data", type=str, required=True,
                    help="Chemin vers training.1600000.processed.noemoticon.csv")
    ap.add_argument("--experiment", type=str, default="use_mlp",
                    help="Nom de l'expérience MLflow")
    ap.add_argument("--subset_rows", type=int, default=200000,
                    help="Sous-échantillon équilibré (total). None = full dataset.")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args()

# ------------------------------------------------------------------
# Chargement équilibré (évite l’échantillon mono-classe)
# ------------------------------------------------------------------
READ_KW = dict(
    header=None,
    names=["target","ids","date","flag","user","text"],
    sep=",", encoding="ISO-8859-1", quotechar='"', engine="python"
)

def load_balanced_subset(path, per_class, chunksize=100_000, seed=42):
    want_neg = per_class     # target 0 -> label 1 (négatif)
    want_pos = per_class     # target 4 -> label 0 (non-négatif)
    neg_parts, pos_parts = [], []

    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        chunk["label"] = chunk["target"].map({0:1, 4:0}).astype(int)
        cneg = chunk[chunk["label"] == 1][["text","label"]]
        cpos = chunk[chunk["label"] == 0][["text","label"]]
        if want_neg > 0 and len(cneg):
            take = min(want_neg, len(cneg))
            neg_parts.append(cneg.sample(n=take, random_state=seed))
            want_neg -= take
        if want_pos > 0 and len(cpos):
            take = min(want_pos, len(cpos))
            pos_parts.append(cpos.sample(n=take, random_state=seed))
            want_pos -= take
        if want_neg <= 0 and want_pos <= 0:
            break

    dfb = pd.concat(neg_parts + pos_parts, ignore_index=True)
    return dfb.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ------------------------------------------------------------------
# Utilitaires MLflow
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

def log_confusion_and_roc(y_true, y_pred, y_score):
    # confusion
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

    # roc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("auc", float(roc_auc))
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

# ------------------------------------------------------------------
# Modèle USE + MLP
# ------------------------------------------------------------------
def build_use_mlp(lr=1e-3):
    use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                               input_shape=[], dtype=tf.string, trainable=False, name="USE")
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = use_layer(inputs)             # (batch, 512)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # MLflow init
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # Data
    if args.subset_rows:
        df = load_balanced_subset(args.data, per_class=args.subset_rows // 2,
                                  chunksize=100_000, seed=args.random_state)
    else:
        df = pd.read_csv(args.data, **READ_KW)
        df["label"] = df["target"].map({0:1, 4:0}).astype(int)
        df = df[["text","label"]]

    # garde-fou
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise RuntimeError("Échantillon mono-classe. Relance avec subset équilibré.")

    X = df["text"].astype(str).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    run_name = f"use_mlp_e{args.epochs}_bs{args.batch_size}"
    with mlflow.start_run(run_name=run_name):
        # tags git
        for k, v in git_info().items():
            if v: mlflow.set_tag(k, v)

        # params
        mlflow.log_params({
            "model": "USE+MLP",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "subset_rows": args.subset_rows,
            "test_size": args.test_size
        })

        # modèle
        model = build_use_mlp(lr=args.lr)

        # callbacks simples
        ckpt_dir = "models/use_mlp"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "weights.keras")
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
        ]

        # entraînement
        t0 = time.perf_counter()
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=cbs
        )
        dur = time.perf_counter() - t0
        mlflow.log_metric("duration_sec", float(dur))

        # log courbes d'apprentissage
        fig = plt.figure()
        plt.plot(hist.history["accuracy"], label="acc")
        plt.plot(hist.history["val_accuracy"], label="val_acc")
        plt.title("Training curves"); plt.legend()
        mlflow.log_figure(fig, "learning_curves.png"); plt.close(fig)

        # évaluation + métriques custom
        y_prob = model.predict(X_test, batch_size=args.batch_size).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("accuracy", float(acc))

        # plots confusion + ROC
        log_confusion_and_roc(y_test, y_pred, y_prob)

        # rapport texte
        rep = classification_report(y_test, y_pred, digits=3)
        with open("classification_report.txt", "w") as f:
            f.write(rep)
        mlflow.log_artifact("classification_report.txt")

        # sauvegarde modèle (SavedModel)
        save_dir = "models/use_mlp_savedmodel"
        model.save(save_dir, include_optimizer=False)
        mlflow.log_artifact(ckpt_path)
        mlflow.log_artifact(save_dir)

        print(f"✅ USE+MLP — F1_macro: {f1:.4f} | acc: {acc:.4f} | dur: {dur:.1f}s")

if __name__ == "__main__":
    main()
