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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    ConfusionMatrixDisplay
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
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
    model.eval()
    losses, preds, labels = [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        out = model(input_ids, attention_mask=attention_mask, labels=y)
        loss, logits = out.loss, out.logits

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    return np.mean(losses), f1_score(y_true, y_pred, average="macro"), accuracy_score(y_true, y_pred), y_true, y_pred


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--subset_rows", type=int, default=200000,
                        help="Sous-échantillon équilibré si renseigné (≈ moitié nég/pos).")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="bert_finetune")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # MLflow local
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.exp_name)

    # Chargement données
    cols = ["target","ids","date","flag","user","text"]
    df = pd.read_csv(
        args.data, header=None, names=cols, encoding="ISO-8859-1",
        engine="python", on_bad_lines="skip"
    )
    # Mapping aligné: 0 = négatif (target=0), 1 = positif (target=4)
    df["label"] = df["target"].map({0:0, 4:1}).astype(int)
    df = df[["text","label"]].dropna()

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

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    with mlflow.start_run(run_name=f"{args.model_name}_finetune"):
        mlflow.log_params({
            **vars(args),
            "device": str(device),
            "warmup_steps": warmup_steps,
            "total_steps": total_steps
        })
        start = time.time()

        best_val_f1 = -1.0
        best_path = f"models/{args.model_name.replace('/', '_')}"
        os.makedirs(best_path, exist_ok=True)

        last_y_true, last_y_pred = None, None

        for epoch in range(args.epochs):
            tr_loss, tr_f1, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss, val_f1, val_acc, y_true, y_pred = eval_epoch(model, val_loader, device)
            last_y_true, last_y_pred = y_true, y_pred

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train loss {tr_loss:.4f} f1 {tr_f1:.4f} acc {tr_acc:.4f} | "
                  f"Val loss {val_loss:.4f} f1 {val_f1:.4f} acc {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_f1": tr_f1, "train_acc": tr_acc,
                "val_loss": val_loss, "val_f1": val_f1, "val_acc": val_acc
            }, step=epoch+1)

            # Sauvegarde du meilleur modèle (selon F1 val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                # log uniquement le répertoire une seule fois (si tu veux éviter les multiples copies, déplace ce log hors de la boucle)
                mlflow.log_artifacts(best_path)

        dur = time.time() - start
        mlflow.log_metric("duration", dur)

        # Artefacts : confusion matrix + classification report (sur la dernière éval)
        if last_y_true is not None and last_y_pred is not None:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(last_y_true, last_y_pred, ax=ax)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            with open("confusion_val.png", "wb") as f:
                f.write(buf.getvalue())
            mlflow.log_artifact("confusion_val.png")

            with open("classification_report.txt", "w") as f:
                f.write(classification_report(last_y_true, last_y_pred, digits=4))
            mlflow.log_artifact("classification_report.txt")

        print(f"✅ {args.model_name} — best Val F1_macro: {best_val_f1:.4f} | dur: {dur:.1f}s")


if __name__ == "__main__":
    main()