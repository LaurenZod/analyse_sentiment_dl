# scripts/run_bert.py
# Fine-tuning DistilBERT en PyTorch pur + logging MLflow (MPS/Apple ok, warmup, clipping, artefacts)

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"        # force Transformers en mode PyTorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import argparse
import random
import io
import numpy as np
import pandas as pd
import mlflow
import mlflow.transformers 
from mlflow.tracking import MlflowClient
from mlflow import transformers as mlflow_hf
import contextlib

# Avant tout usage MLflow :
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")      # 5 min
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "8")    # <= 8 pour éviter l’erreur “in excess of maximum”
os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")    # au cas où HTTPS est absent
os.environ.setdefault("MLFLOW_AUTOLOGGING_ENABLED", "false")     # pas d’autolog TF/Keras qui spam les endpoints

# Helper: n’échoue jamais si MLflow est KO
def try_mlflow(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as e:
        print(f"[MLflow WARN] {call.__name__} failed: {e}")
        return None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc   # <-- ajout ROC/AUC
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, pipeline
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------
# Utils
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def balanced_subset(df: pd.DataFrame, total_rows: int, seed: int = 42) -> pd.DataFrame:
    """Prend un sous-échantillon équilibré (moitié négatifs, moitié positifs)."""
    n_per_class = total_rows // 2
    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]
    take_neg = min(n_per_class, len(neg))
    take_pos = min(n_per_class, len(pos))
    dfb = pd.concat([
        neg.sample(n=take_neg, random_state=seed),
        pos.sample(n=take_pos, random_state=seed)
    ])
    return dfb.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------
# Dataset PyTorch custom
# ----------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ----------------------
# Entraînement
# ----------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    losses, preds, labels = [], [], []

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        out = model(input_ids, attention_mask=attention_mask, labels=y)
        loss, logits = out.loss, out.logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())

    return np.mean(losses), f1_score(labels, preds, average="macro"), accuracy_score(labels, preds)


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    """Retourne aussi les probabilités de la classe 1 pour ROC/AUC."""
    model.eval()
    losses, preds, labels, scores = [], [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        out = model(input_ids, attention_mask=attention_mask, labels=y)
        loss, logits = out.loss, out.logits

        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(classe=1)
        scores.extend(probs.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    y_scores = np.array(scores)
    return np.mean(losses), f1_score(y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred), y_true, y_pred, y_scores


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--subset_rows", type=int, default=200000,
                        help="Sous-échantillon équilibré si renseigné (≈ moitié nég/pos).")
    parser.add_argument("--model_name", type=str,
                        default="models/bert/distilbert-base-uncased",
                        help="Nom du modèle (chemin local ou ID Hugging Face, ex: 'distilbert-base-uncased' ou dossier local).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="bert_finetune")
    parser.add_argument("--register_alias", type=str, default="", help="Alias MLflow à appliquer à la version enregistrée (ex: 'tweet_predictions_bert'). Laisser vide pour ne pas créer/modifier d'alias.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # MLflow: respect env var if provided, fallback to local file store
    uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri if uri else "file:./mlruns")
    try_mlflow(mlflow.set_experiment, args.exp_name)

    # Chargement données
    cols = ["target","ids","date","flag","user","text"]
    df = pd.read_csv(
        args.data, header=None, names=cols, encoding="ISO-8859-1",
        engine="python", on_bad_lines="skip"
    )
    # Mapping aligné: 0 = négatif (target=0), 1 = positif (target=4)
    df["label"] = df["target"].map({0:0, 4:1})
    df = df[["text","label"]].dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    assert set(df["label"].unique()) <= {0, 1}
    

    # Sous-échantillon équilibré si demandé
    if args.subset_rows:
        df = balanced_subset(df, total_rows=args.subset_rows, seed=args.seed)

    train_texts, val_texts, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=args.seed
    )

    # Tokenizer / Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = TweetDataset(train_texts, y_train, tokenizer, args.max_length)
    val_ds   = TweetDataset(val_texts, y_val, tokenizer, args.max_length)

    # DataLoaders (pin_memory utile uniquement en CUDA)
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=pin_mem)

    # Modèle
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    no_improve = 0
    best_epoch = -1

    with mlflow.start_run(run_name=f"{args.model_name}_finetune") as run:
        try_mlflow(mlflow.log_params, {
            **vars(args),
            "device": str(device),
            "warmup_steps": warmup_steps,
            "total_steps": total_steps
        })
        start = time.time()

        best_path = os.path.join("models", "bert", args.model_name.replace("/", "_"))
        os.makedirs(best_path, exist_ok=True)

        last_y_true, last_y_pred, last_y_scores = None, None, None

        for epoch in range(args.epochs):
            tr_loss, tr_f1, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_out = eval_epoch(model, val_loader, device)
            val_loss, val_f1, val_acc, y_true, y_pred, y_scores = val_out
            last_y_true, last_y_pred, last_y_scores = y_true, y_pred, y_scores

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train loss {tr_loss:.4f} f1 {tr_f1:.4f} acc {tr_acc:.4f} | "
                  f"Val loss {val_loss:.4f} f1 {val_f1:.4f} acc {val_acc:.4f}")

            try_mlflow(mlflow.log_metrics, {
                "train_loss": tr_loss, "train_f1": tr_f1, "train_acc": tr_acc,
                "val_loss": val_loss, "val_f1": val_f1, "val_acc": val_acc
            }, step=epoch+1)

            if val_f1 > best_val_f1 + 1e-4:   
                best_val_f1 = val_f1
                no_improve = 0
                best_epoch = epoch + 1
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"Early stopping (patience={args.early_stop_patience}). Best val_f1={best_val_f1:.4f}")
                    break


        dur = time.time() - start
        try_mlflow(mlflow.log_metric, "duration", dur)
        try_mlflow(mlflow.log_metric, "best_val_f1", float(best_val_f1))
        try_mlflow(mlflow.log_metric, "best_epoch", int(best_epoch))

        # Artefacts : confusion matrix + classification report + ROC (sur la dernière éval)
        if last_y_true is not None and last_y_pred is not None:
            # Confusion matrix figure -> directly to MLflow
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(last_y_true, last_y_pred, ax=ax)
            try_mlflow(mlflow.log_figure, fig, "confusion_val.png")
            plt.close(fig)

            # Classification report -> directly as text artifact
            rep_txt = classification_report(last_y_true, last_y_pred, digits=4)
            try_mlflow(mlflow.log_text, rep_txt, "classification_report.txt")

            # ROC + AUC (needs probability scores)
            if last_y_scores is not None:
                fpr, tpr, _ = roc_curve(last_y_true, last_y_scores)
                roc_auc = auc(fpr, tpr)
                try_mlflow(mlflow.log_metric, "auc", float(roc_auc))

                fig = plt.figure()
                plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve"); plt.legend()
                try_mlflow(mlflow.log_figure, fig, "roc_curve.png")
                plt.close(fig)

        try_mlflow(mlflow.set_tags, {"framework": "transformers", "hf_model": args.model_name, "device": str(device)})

        # Recharge le meilleur modèle et log vers MLflow + Model Registry
        pipe = pipeline(
            task="text-classification",
            model=best_path,
            tokenizer=best_path,
            device=-1  # CPU pour la sérialisation
        )

        # Log du modèle au format MLflow + enregistrement dans le Model Registry
        try:
            model_info = mlflow_hf.log_model(
                transformers_model=pipe,
                artifact_path="bert_model",
                task="text-classification",
                registered_model_name="tweet_prediction",
            )
            print(f"[MLflow] Modèle loggué dans les artefacts : {model_info.model_uri}")
        except Exception as e:
            print(f"[MLflow WARN] log_model failed: {e}")
            model_info = None

        # Alias optionnel (si on connaît la version créée)
        if args.register_alias:
            try:
                client = MlflowClient()
                created_version = None
                if model_info is not None:
                    # retrouver la version dont la source correspond au model_uri loggué
                    for mv in client.get_latest_versions("tweet_prediction"):
                        if mv.source == model_info.model_uri:
                            created_version = mv.version
                            break

                if created_version is not None:
                    client.set_registered_model_alias(
                        name="tweet_prediction",
                        alias=args.register_alias,
                        version=created_version,
                    )
                    print(f"[MLflow] Alias '{args.register_alias}' -> v{created_version}")
                else:
                    print("[MLflow WARN] Impossible de retrouver la version créée pour poser l'alias.")
            except Exception as e:
                print(f"[MLflow WARN] set alias failed: {e}")

        print(f"✅ {args.model_name} — best val_f1: {best_val_f1:.4f} | dur: {dur:.1f}s")


if __name__ == "__main__":
    main()