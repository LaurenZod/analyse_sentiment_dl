# scripts/run_textcnn.py (version optimisée)
import os, re, time, argparse, subprocess, io
import numpy as np
import pandas as pd
import mlflow, mlflow.tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="TextCNN (Conv1D) + embeddings pré-entraînés, log MLflow (version optimisée)")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--experiment", type=str, default="textcnn_embeddings")
    ap.add_argument("--subset_rows", type=int, default=200000)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # TextVectorization
    ap.add_argument("--max_tokens", type=int, default=100000)   # ↑
    ap.add_argument("--seq_len", type=int, default=80)          # ↑

    # Embedding
    ap.add_argument("--embedding_path", type=str, required=True)
    ap.add_argument("--embedding_dim", type=int, required=True)
    ap.add_argument("--trainable_embed", action="store_true", help="rendre l’embedding entraînable dès le départ")

    # CNN / régularisation
    ap.add_argument("--filters", type=int, default=256)         # ↑
    ap.add_argument("--kernel_sizes", type=str, default="3,4,5")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--spatial_dropout", type=float, default=0.2)  # ↑

    # Entraînement
    ap.add_argument("--epochs", type=int, default=10)           # ↑
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Fine-tuning embedding (optionnel si non-trainable au début)
    ap.add_argument("--ft_epochs", type=int, default=2)
    ap.add_argument("--ft_lr", type=float, default=5e-4)

    return ap.parse_args()

# -------------- Chargement data --------------
READ_KW = dict(
    header=None, names=["target","ids","date","flag","user","text"],
    sep=",", encoding="ISO-8859-1", quotechar='"', engine="python"
)

def load_balanced_subset(path, per_class, chunksize=100_000, seed=42):
    want_neg = per_class  # target 0 -> label 1
    want_pos = per_class  # target 4 -> label 0
    neg_parts, pos_parts = [], []

    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        chunk["label"] = chunk["target"].map({0:1, 4:0}).astype(int)
        cneg = chunk[chunk["label"] == 1][["text","label"]]
        cpos = chunk[chunk["label"] == 0][["text","label"]]
        if want_neg > 0 and len(cneg):
            take = min(want_neg, len(cneg)); neg_parts.append(cneg.sample(n=take, random_state=seed)); want_neg -= take
        if want_pos > 0 and len(cpos):
            take = min(want_pos, len(cpos)); pos_parts.append(cpos.sample(n=take, random_state=seed)); want_pos -= take
        if want_neg <= 0 and want_pos <= 0:
            break

    dfb = pd.concat(neg_parts + pos_parts, ignore_index=True)
    return dfb.sample(frac=1.0, random_state=seed).reset_index(drop=True)

CLEAN_URL  = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
CLEAN_USER = re.compile(r'@\w+')
CLEAN_HASH = re.compile(r'#(\w+)')
CLEAN_WS   = re.compile(r'\s+')

def basic_clean(t: str) -> str:
    t = CLEAN_URL.sub(' ', str(t))
    t = CLEAN_USER.sub(' ', t)
    t = CLEAN_HASH.sub(r'\1', t)
    t = CLEAN_WS.sub(' ', t)
    return t.strip().lower()

# ----------- Embedding loader (.txt) -----------
def load_embedding_txt(path, embedding_dim, vocab):
    vocab_to_index = {t: i for i, t in enumerate(vocab)}
    matrix = np.random.normal(scale=0.01, size=(len(vocab), embedding_dim)).astype(np.float32)
    if vocab and vocab[0] == "":  # pad
        matrix[0] = 0.0

    found = 0
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < embedding_dim + 1:
                continue
            word = parts[0]
            idx = vocab_to_index.get(word)
            if idx is not None:
                try:
                    vec = np.asarray(parts[-embedding_dim:], dtype="float32")
                except Exception:
                    continue
                matrix[idx] = vec
                found += 1
    return matrix, found

# -------------- Modèle TextCNN --------------
def build_textcnn(vocab_size, seq_len, embedding_matrix, trainable_embed, filters, kernel_sizes, dropout, spatial_dropout, lr=1e-3):
    inp = keras.Input(shape=(seq_len,), dtype="int32")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=trainable_embed,
        mask_zero=False,
        name="embedding"
    )(inp)

    # plus robuste au sur-apprentissage des séquences
    if spatial_dropout and spatial_dropout > 0:
        x = keras.layers.SpatialDropout1D(spatial_dropout)(x)

    branches = []
    for k in kernel_sizes:
        b = keras.layers.Conv1D(filters=filters, kernel_size=k, padding="valid", activation="relu")(x)
        b = keras.layers.GlobalMaxPooling1D()(b)
        branches.append(b)
    x = keras.layers.Concatenate()(branches)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------- Utils MLflow --------------
def git_info():
    def _git(args):
        try: return subprocess.check_output(["git"]+args, text=True).strip()
        except Exception: return None
    return {
        "git.commit": _git(["rev-parse", "--short", "HEAD"]),
        "git.branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git.remote": _git(["config", "--get", "remote.origin.url"]),
    }

def log_plots(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion matrix"); plt.colorbar()
    ticks = np.arange(2); plt.xticks(ticks, ["non_neg","neg"]); plt.yticks(ticks, ["non_neg","neg"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Pred"); plt.ylabel("True")
    mlflow.log_figure(fig, "confusion_matrix.png"); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr); mlflow.log_metric("auc", float(roc_auc))
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    mlflow.log_figure(fig, "roc_curve.png"); plt.close(fig)

# -------------- Main --------------
def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # Data
    if args.subset_rows:
        df = load_balanced_subset(args.data, per_class=args.subset_rows // 2, chunksize=100_000, seed=args.random_state)
    else:
        df = pd.read_csv(args.data, **READ_KW)
        df["label"] = df["target"].map({0:1, 4:0}).astype(int)
        df = df[["text","label"]]

    df["text"] = df["text"].astype(str).apply(basic_clean)
    X = df["text"].values
    y = df["label"].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=args.random_state, stratify=y)

    # Vectorizer
    vectorizer = keras.layers.TextVectorization(
        max_tokens=args.max_tokens, output_mode="int", output_sequence_length=args.seq_len,
        standardize=None, split="whitespace"
    )
    vectorizer.adapt(X_train)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    # Embedding matrix
    emb_matrix, found = load_embedding_txt(args.embedding_path, args.embedding_dim, vocab)
    print(f"Vocab size: {vocab_size} | Found {found} embeddings from {os.path.basename(args.embedding_path)}")

    # ids tensors
    Xtr_ids = vectorizer(X_train); Xte_ids = vectorizer(X_test)

    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",")]

    run_name = f"textcnn_opt_{os.path.basename(args.embedding_path)}_{args.embedding_dim}d_len{args.seq_len}_tok{args.max_tokens}"
    with mlflow.start_run(run_name=run_name):
        for k, v in git_info().items():
            if v: mlflow.set_tag(k, v)

        mlflow.log_params({
            "model": "TextCNN(opt)",
            "embedding_file": os.path.basename(args.embedding_path),
            "embedding_dim": args.embedding_dim,
            "trainable_embed": args.trainable_embed,
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "filters": args.filters,
            "kernel_sizes": args.kernel_sizes,
            "dropout": args.dropout,
            "spatial_dropout": args.spatial_dropout,
            "subset_rows": args.subset_rows
        })

        # Build
        model = build_textcnn(vocab_size, args.seq_len, emb_matrix, args.trainable_embed,
                              args.filters, kernel_sizes, args.dropout, args.spatial_dropout, lr=args.lr)

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ]

        t0 = time.perf_counter()
        hist = model.fit(Xtr_ids, y_train,
                         validation_data=(Xte_ids, y_test),
                         epochs=args.epochs, batch_size=args.batch_size,
                         callbacks=cbs, verbose=1)
        dur = time.perf_counter() - t0
        mlflow.log_metric("duration_sec", float(dur))

        # Courbes
        fig = plt.figure()
        plt.plot(hist.history["accuracy"], label="acc")
        plt.plot(hist.history["val_accuracy"], label="val_acc")
        plt.title("Training curves"); plt.legend()
        mlflow.log_figure(fig, "learning_curves.png"); plt.close(fig)

        # Eval 1
        y_prob = model.predict(Xte_ids, batch_size=args.batch_size).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred, average="macro"); acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("f1_macro", float(f1)); mlflow.log_metric("accuracy", float(acc))
        log_plots(y_test, y_pred, y_prob)

        # Optionnel : fine-tuning embedding si non trainable au départ
        if not args.trainable_embed and args.ft_epochs > 0:
            model.get_layer("embedding").trainable = True
            model.compile(optimizer=keras.optimizers.Adam(args.ft_lr), loss="binary_crossentropy", metrics=["accuracy"])
            hist_ft = model.fit(Xtr_ids, y_train,
                                validation_data=(Xte_ids, y_test),
                                epochs=args.ft_epochs, batch_size=args.batch_size, verbose=1)
            y_prob = model.predict(Xte_ids, batch_size=args.batch_size).ravel()
            y_pred = (y_prob >= 0.5).astype(int)
            f1_ft = f1_score(y_test, y_pred, average="macro"); acc_ft = accuracy_score(y_test, y_pred)
            mlflow.log_metric("f1_macro_finetune", float(f1_ft)); mlflow.log_metric("accuracy_finetune", float(acc_ft))

        # Rapport + sauvegardes
        rep = classification_report(y_test, y_pred, digits=3)
        with open("classification_report.txt", "w") as f: f.write(rep)
        mlflow.log_artifact("classification_report.txt")

        os.makedirs("models/textcnn", exist_ok=True)
        model.save("models/textcnn/model.keras", include_optimizer=False)
        model.export("models/textcnn/saved_model")
        with open("models/textcnn/vocab.txt", "w") as f: f.write("\n".join(vocab))
        mlflow.log_artifact("models/textcnn/model.keras")
        mlflow.log_artifact("models/textcnn/vocab.txt")
        mlflow.log_artifacts("models/textcnn/saved_model")

        print(f"✅ TextCNN(opt) — F1_macro: {f1:.4f} | acc: {acc:.4f} | dur: {dur:.1f}s")

if __name__ == "__main__":
    main()
