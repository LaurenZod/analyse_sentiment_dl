# scripts/run_lstm_torch.py
# LSTM + GloVe (PyTorch pur) avec tracking MLflow

import os
import re
import time
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import mlflow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Prétraitement minimal
# -----------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
USER_RE = re.compile(r'@\w+')
HASH_RE = re.compile(r'#(\w+)')
MULTIWS = re.compile(r'\s+')
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"

def normalize_text(t: str) -> str:
    t = URL_RE.sub(' ', t)
    t = USER_RE.sub(' ', t)
    t = HASH_RE.sub(r'\1', t)
    t = MULTIWS.sub(' ', t).strip()
    return t.lower()

def tokenize(t: str):
    return WORD_RE.findall(t)  # simple découpe alphanum/apostrophe


# -----------------------
# Vocab & séquences
# -----------------------
def build_vocab(texts, max_tokens=80000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    # tokens triés par fréquence
    vocab = [PAD_TOKEN, OOV_TOKEN] + [w for w, c in counter.most_common() if c >= min_freq]
    if len(vocab) > max_tokens:
        vocab = vocab[:max_tokens]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos

def text_to_ids(text, stoi, seq_len):
    toks = tokenize(text)
    ids = [stoi.get(tok, stoi[OOV_TOKEN]) for tok in toks]
    if len(ids) >= seq_len:
        return ids[:seq_len]
    # pad
    return ids + [stoi[PAD_TOKEN]] * (seq_len - len(ids))


# -----------------------
# Chargement GloVe
# -----------------------
def load_glove_matrix(path_txt: str, embedding_dim: int, stoi: dict):
    embedding_matrix = np.random.normal(scale=0.6, size=(len(stoi), embedding_dim)).astype("float32")
    # PAD = 0 vector
    embedding_matrix[stoi[PAD_TOKEN]] = np.zeros((embedding_dim,), dtype="float32")
    found = 0
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = parts[1:]
            if len(vec) != embedding_dim:
                continue
            if word in stoi:
                embedding_matrix[stoi[word]] = np.asarray(vec, dtype="float32")
                found += 1
    return embedding_matrix, found


# -----------------------
# Dataset PyTorch
# -----------------------
class SeqDataset(Dataset):
    def __init__(self, texts, labels, stoi, seq_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.stoi = stoi
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = text_to_ids(self.texts[i], self.stoi, self.seq_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.float32)


# -----------------------
# Modèle LSTM
# -----------------------
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=128, bidirectional=False,
                 dropout=0.4, pad_idx=0, embedding_matrix=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embed

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0  # dropout inter-couches s'applique si num_layers>1
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, 1)  # binaire

    def forward(self, x):
        # x: (batch, seq_len) avec le padding_idx défini dans l'embedding
        pad_idx = self.embedding.padding_idx
        # longueurs réelles (nb de tokens ≠ PAD) ; pack() exige des longueurs sur CPU
        lengths = (x != pad_idx).sum(dim=1).to("cpu").clamp_min(1)

        emb = self.embedding(x)  # (B, T, E)

        # On "packe" les séquences pour que le LSTM ignore les pas de PAD
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, H)

        # On prend l'état final de la dernière couche (concat des 2 directions si bidirectionnel)
        if self.lstm.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            feat = h_n[-1]  # (B, H)

        feat = self.dropout(feat)
        logits = self.fc(feat).squeeze(-1)  # (B,)
        return logits  # BCEWithLogitsLoss attend des logits (pas une sigmoïde)

# -----------------------
# Confusion matrix plot
# -----------------------
def log_confusion_png(y_true, y_prob, out_png):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    classes = ["negatif", "positif"]
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=classes, yticklabels=classes,
           ylabel="Vrai", xlabel="Prédit", title="Matrice de confusion")
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(out_png)


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    losses, preds, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().numpy())
        labels.extend(y.long().cpu().numpy())
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return float(np.mean(losses)), float(f1), float(acc)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, labels, probs = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        p = torch.sigmoid(logits)
        losses.append(loss.item())
        preds.extend((p >= 0.5).long().cpu().numpy())
        labels.extend(y.long().cpu().numpy())
        probs.extend(p.cpu().numpy())
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return float(np.mean(losses)), float(f1), float(acc), np.asarray(labels), np.asarray(probs).ravel()


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--subset_rows", type=int, default=200000)
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Chemin du GloVe (ex: glove.twitter.27B.200d.txt)")
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=80000)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_embed_epochs", type=int, default=2,
                        help="Nb d'epochs avec embeddings gelés avant de les dégeler")
    args = parser.parse_args()

    mlflow.set_experiment("lstm_embeddings_torch")

    # Device (MPS Apple si dispo)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    # 1) Données
    cols = ["target","ids","date","flag","user","text"]
    df = pd.read_csv(args.data, header=None, names=cols, encoding="ISO-8859-1")
    df["label"] = df["target"].map({0:0, 4:1})
    df = df[["text","label"]].dropna()
    if args.subset_rows:
        df = df.sample(n=args.subset_rows, random_state=42)
    df["text_clean"] = df["text"].astype(str).apply(normalize_text)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        df["text_clean"], df["label"].astype(int),
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 2) Vocab
    stoi, itos = build_vocab(X_train, max_tokens=args.max_tokens, min_freq=1)
    print(f"Vocab size: {len(stoi)}")
    # 3) Embedding matrix
    emb_matrix, found = load_glove_matrix(args.embedding_path, args.embedding_dim, stoi)
    print(f"Found {found} words in {os.path.basename(args.embedding_path)}")

    # 4) Dataset/Loader
    tr_ds = SeqDataset(X_train, y_train.values, stoi, args.seq_len)
    va_ds = SeqDataset(X_val, y_val.values, stoi, args.seq_len)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5) Modèle
    model = LSTMSentiment(
        vocab_size=len(stoi),
        embedding_dim=args.embedding_dim,
        hidden_size=128,
        bidirectional=args.bidirectional,
        dropout=0.4,
        pad_idx=stoi[PAD_TOKEN],
        embedding_matrix=emb_matrix,
        freeze_embed=True  # d'abord gelé
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name=f"LSTM_torch_glove_{args.embedding_dim}d"):
        mlflow.log_params({
            "subset_rows": args.subset_rows,
            "embedding_dim": args.embedding_dim,
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "bidirectional": args.bidirectional,
            "lr": args.lr,
            "freeze_embed_epochs": args.freeze_embed_epochs,
            "embedding_path": args.embedding_path
        })

        start = time.time()
        for epoch in range(1, args.epochs + 1):
            # dégel des embeddings après n epochs
            if epoch == args.freeze_embed_epochs + 1:
                model.embedding.weight.requires_grad = True
                # baisse du LR quand on fine-tune l'embedding
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * 0.5

            tr_loss, tr_f1, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
            va_loss, va_f1, va_acc, y_true, y_prob = eval_epoch(model, va_loader, criterion, device)

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train loss {tr_loss:.4f} f1 {tr_f1:.4f} acc {tr_acc:.4f} | "
                  f"Val loss {va_loss:.4f} f1 {va_f1:.4f} acc {va_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_f1": tr_f1, "train_acc": tr_acc,
                "val_loss": va_loss, "val_f1": va_f1, "val_acc": va_acc
            }, step=epoch)

        dur = time.time() - start
        mlflow.log_metric("duration", float(dur))

        # Artifacts : matrice de confusion + rapport
        os.makedirs("artifacts", exist_ok=True)
        cm_png = "artifacts/lstm_torch_confusion.png"
        log_confusion_png(y_true, y_prob, cm_png)
        mlflow.log_text(
            classification_report(y_true, (y_prob>=0.5).astype(int), digits=3),
            "artifacts/lstm_torch_classification_report.txt"
        )

        # Sauvegarde du modèle
        save_dir = "models/lstm_torch"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "lstm_torch.pt")
        torch.save({"state_dict": model.state_dict(),
                    "stoi": stoi,
                    "seq_len": args.seq_len,
                    "embedding_dim": args.embedding_dim}, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ LSTM (PyTorch) — F1_macro: {va_f1:.4f} | acc: {va_acc:.4f} | dur: {dur:.1f}s")


if __name__ == "__main__":
    main()
