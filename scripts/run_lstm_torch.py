# scripts/run_lstm_torch.py
import os, re, io, time, argparse, random, math, subprocess
from collections import Counter

import numpy as np
import pandas as pd
import mlflow
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

# ---------- Utils device / seed ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Light text cleaning ----------
URL = re.compile(r'https?://\S+|www\.\S+')
USER = re.compile(r'@\w+')
HASH = re.compile(r'#(\w+)')
WS   = re.compile(r'\s+')

def normalize_light(t: str) -> str:
    t = URL.sub(' ', str(t))
    t = USER.sub(' ', t)
    t = HASH.sub(r'\1', t)
    t = WS.sub(' ', t).strip().lower()
    return t

# ---------- Data loading ----------
READ_KW = dict(
    header=None, names=["target","ids","date","flag","user","text"],
    sep=",", encoding="ISO-8859-1", quotechar='"', engine="python"
)

def load_balanced_subset_csv(path, per_class, chunksize=100_000, seed=42):
    want_neg = per_class   # target 0  -> label 0 (négatif)
    want_pos = per_class   # target 4  -> label 1 (positif)
    neg_parts, pos_parts = [], []
    for chunk in pd.read_csv(path, chunksize=chunksize, on_bad_lines="skip", **READ_KW):
        # Mapping cohérent: 0 = négatif (target=0), 1 = positif (target=4)
        chunk["label"] = chunk["target"].map({0:0, 4:1}).astype(int)
        cneg = chunk[chunk["label"] == 0][["text","label"]]
        cpos = chunk[chunk["label"] == 1][["text","label"]]
        if want_neg > 0 and len(cneg):
            take = min(want_neg, len(cneg)); neg_parts.append(cneg.sample(n=take, random_state=seed)); want_neg -= take
        if want_pos > 0 and len(cpos):
            take = min(want_pos, len(cpos)); pos_parts.append(cpos.sample(n=take, random_state=seed)); want_pos -= take
        if want_neg <= 0 and want_pos <= 0:
            break
    df = pd.concat(neg_parts + pos_parts, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ---------- Vocab / vectorization ----------
PAD, UNK = "<pad>", "<unk>"

def build_vocab(texts, max_tokens=80_000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    # garde tokens par fréquence, + PAD/UNK en tête
    vocab = [PAD, UNK] + [w for w, c in counter.most_common(max_tokens-2) if c >= min_freq]
    stoi = {w:i for i, w in enumerate(vocab)}
    return vocab, stoi

def texts_to_ids(texts, stoi, seq_len):
    ids = []
    unk = stoi[UNK]
    for t in texts:
        toks = t.split()
        row = [stoi.get(w, unk) for w in toks[:seq_len]]
        if len(row) < seq_len:
            row += [0]*(seq_len-len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)

# ---------- Load GloVe into matrix ----------
def load_glove_txt(path, embedding_dim, vocab):
    vocab_to_index = {t:i for i,t in enumerate(vocab)}
    matrix = np.random.normal(scale=0.01, size=(len(vocab), embedding_dim)).astype(np.float32)
    if vocab and vocab[0] == PAD:
        matrix[0] = 0.0
    found = 0
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < embedding_dim + 1:
                continue
            w = parts[0]
            idx = vocab_to_index.get(w)
            if idx is not None:
                try:
                    vec = np.asarray(parts[-embedding_dim:], dtype="float32")
                except Exception:
                    continue
                matrix[idx] = vec
                found += 1
    return torch.tensor(matrix), found

# ---------- Dataset ----------
class SeqDataset(Dataset):
    def __init__(self, X_ids, y):
        self.X = X_ids
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ---------- Model ----------
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden=128, bidirectional=True, dropout=0.3, freeze_embed_epochs=0):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = (freeze_embed_epochs == 0)

        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden, batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def unfreeze_embedding(self):
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        # x: [B, T] with PAD id = 0
        # Compute true sequence lengths (>=1)
        with torch.no_grad():
            lengths = (x != 0).sum(dim=1).clamp(min=1)
        # Embed
        x_emb = self.embedding(x)  # [B, T, E]
        # Pack so LSTM ignores PAD tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        # "h" already corresponds to the last valid timestep per sequence
        if self.lstm.bidirectional:
            # Concatenate last hidden states from both directions (last layer)
            rep = torch.cat([h[-2], h[-1]], dim=1)  # [B, 2*H]
        else:
            rep = h[-1]  # [B, H]
        z = self.dropout(rep)
        logits = self.fc(z).squeeze(1)  # [B]
        return logits

# ---------- Git tags for MLflow ----------
def git_info():
    def _git(args):
        try: return subprocess.check_output(["git"]+args, text=True).strip()
        except Exception: return None
    return {
        "git.commit": _git(["rev-parse", "--short", "HEAD"]),
        "git.branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git.remote": _git(["config", "--get", "remote.origin.url"]),
    }

def log_confusion_and_roc(y_true, y_prob):
    # Confusion matrix (threshold 0.5)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("auc", float(roc_auc))
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

# ---------- Train / Eval ----------
def run_epoch(model, loader, criterion, optimizer=None, device=None, return_scores=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, preds, labels = [], [], []
    scores = [] if return_scores else None

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        logits = model(Xb)
        loss = criterion(logits, yb)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).long().cpu().numpy()
        if return_scores:
            # store raw probabilities for ROC
            scores_batch = probs.cpu().numpy().ravel().tolist()
        preds.extend(pred)
        labels.extend(yb.long().cpu().numpy())
        if return_scores:
            scores.extend(scores_batch)

    preds = np.array(preds); labels = np.array(labels)
    acc = (preds == labels).mean()
    # F1 macro binaire
    tp = ((preds==1)&(labels==1)).sum()
    fp = ((preds==1)&(labels==0)).sum()
    fn = ((preds==0)&(labels==1)).sum()
    tn = ((preds==0)&(labels==0)).sum()
    def f1(p, r):
        return 2*p*r/(p+r) if (p+r)>0 else 0.0
    # classe 0
    p0 = tn/ (tn+fn) if (tn+fn)>0 else 0.0
    r0 = tn/ (tn+fp) if (tn+fp)>0 else 0.0
    f10 = f1(p0, r0)
    # classe 1
    p1 = tp/ (tp+fp) if (tp+fp)>0 else 0.0
    r1 = tp/ (tp+fn) if (tp+fn)>0 else 0.0
    f11 = f1(p1, r1)
    f1m = (f10+f11)/2
    if return_scores:
        return float(np.mean(losses)), float(f1m), float(acc), labels.tolist(), scores
    return float(np.mean(losses)), float(f1m), float(acc)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="LSTM (PyTorch, MPS-friendly) + GloVe + MLflow")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--subset_rows", type=int, default=20000)
    ap.add_argument("--embedding_path", type=str, required=True)
    ap.add_argument("--embedding_dim", type=int, required=True)
    ap.add_argument("--max_tokens", type=int, default=60000)
    ap.add_argument("--seq_len", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--freeze_embed_epochs", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=1)
    ap.add_argument("--experiment", type=str, default="lstm_embeddings_torch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # (Optionnel) améliore la précision matmul (utile MPS)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ---------- Data
    if args.subset_rows:
        df = load_balanced_subset_csv(args.data, per_class=args.subset_rows//2, chunksize=100_000, seed=args.seed)
    else:
        df = pd.read_csv(args.data, on_bad_lines="skip", **READ_KW)
        # Mapping cohérent: 0 = négatif, 1 = positif
        df["label"] = df["target"].map({0:0, 4:1}).astype(int)
        df = df[["text","label"]].sample(frac=1.0, random_state=args.seed)

    df["text"] = df["text"].astype(str).apply(normalize_light)
    # split
    n = len(df); n_val = int(0.2*n)
    df_train = df.iloc[:-n_val]; df_val = df.iloc[-n_val:]

    # ---------- Vocab & vectorize
    vocab, stoi = build_vocab(df_train["text"].tolist(), max_tokens=args.max_tokens, min_freq=1)
    Xtr = texts_to_ids(df_train["text"].tolist(), stoi, args.seq_len)
    Xva = texts_to_ids(df_val["text"].tolist(),   stoi, args.seq_len)
    ytr = df_train["label"].astype(int).values
    yva = df_val["label"].astype(int).values

    # ---------- Embeddings
    emb_matrix, found = load_glove_txt(args.embedding_path, args.embedding_dim, vocab)
    print(f"Vocab size: {len(vocab)} | Found {found} embeddings from {os.path.basename(args.embedding_path)}")

    # ---------- Datasets / Loaders
    train_ds = SeqDataset(Xtr, ytr)
    val_ds   = SeqDataset(Xva, yva)

    # pin_memory = False sur MPS; True sur CUDA
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin_mem)

    # ---------- Model
    model = LSTMClassifier(
        embedding_matrix=emb_matrix,
        hidden=128,
        bidirectional=args.bidirectional,
        dropout=0.3,
        freeze_embed_epochs=args.freeze_embed_epochs
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---------- MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"lstm_torch_glove_{args.embedding_dim}d"):
        run_id = mlflow.active_run().info.run_id
        model_dir = os.path.join("models", "lstm_torch", run_id)
        os.makedirs(model_dir, exist_ok=True)
        # tags git
        for k, v in git_info().items():
            if v: mlflow.set_tag(k, v)

        mlflow.log_params({
            **vars(args),
            "device": str(device),
            "vocab_size": len(vocab),
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
        })

        t0 = time.perf_counter()
        best_f1 = -1.0
        best_epoch = 0
        patience = int(getattr(args, "early_stop_patience", 1))
        stalled = 0

        for epoch in range(1, args.epochs+1):
            # (dé)gèle l’embedding après N epochs
            if epoch == args.freeze_embed_epochs + 1:
                model.unfreeze_embedding()

            tr_loss, tr_f1, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_f1, va_acc, y_true, y_prob = run_epoch(model, val_loader, criterion, None, device, return_scores=True)

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train loss {tr_loss:.4f} f1 {tr_f1:.4f} acc {tr_acc:.4f} | "
                  f"Val loss {va_loss:.4f} f1 {va_f1:.4f} acc {va_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_f1": tr_f1, "train_acc": tr_acc,
                "val_loss": va_loss, "val_f1": va_f1, "val_acc": va_acc
            }, step=epoch)

            if epoch == 1 or va_f1 >= best_f1:
                # log diagnostic plots for the current best (or first epoch)
                log_confusion_and_roc(y_true, y_prob)

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_epoch = epoch
                stalled = 0
                torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))
            else:
                stalled += 1
                if stalled >= patience:
                    print(f"Early stopping triggered (no val_f1 improvement for {patience} epoch(s)). "
                          f"Best at epoch {best_epoch} with val_f1={best_f1:.4f}")
                    break

        mlflow.log_artifacts(model_dir, artifact_path="model")
        mlflow.log_metric("best_epoch", float(best_epoch))

        dur = time.perf_counter() - t0
        mlflow.log_metric("duration_sec", float(dur))
        print(f"✅ LSTM (PyTorch/MPS) — best Val F1_macro: {best_f1:.4f} | dur: {dur:.1f}s")

if __name__ == "__main__":
    main()