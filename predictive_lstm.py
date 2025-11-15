# predictive_lstm.py
import math
import matplotlib.pyplot as plt
import random
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm

# ===========================
# Config
# ===========================
# 1) Set your dataset path here (first path tried)
DATA_PATH = Path(r"C:\Users\KIIT\Desktop\major_project _2\wikitext2_train.txt")
# 2) Fallback (works if you kept the uploaded file in this location)
FALLBACK_PATH = Path("wikitext2_train.txt")  # or Path("/mnt/data/wikitext2_train.txt")

VOCAB_SIZE = 8000
CONTEXT_SIZE = 6
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
EMB_DIM = 128
HIDDEN_DIM = 192
NUM_LAYERS = 2


# ===========================
# Data loading / cleaning
# ===========================
def load_corpus(path: Path, min_len: int = 10, max_len: int = 300) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines: List[str] = []
    for ln in text:
        ln = ln.strip()
        if not ln:
            continue
        # Simple cleanup
        ln = re.sub(r"\s+", " ", ln)
        ln = re.sub(r"\[[^\]]+\]", "", ln)
        if min_len <= len(ln) <= max_len:
            lines.append(ln)
    return lines


# ===========================
# Dataset + Tokenizer helpers
# ===========================
class SimpleTextDataset(Dataset):
    """
    Dataset for next-word prediction.
    Each item is (context_indices, target_index).
    """
    def __init__(self, texts: List[str], tokenizer: Tokenizer, context_size: int = 6):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.examples = []
        for t in texts:
            enc = tokenizer.encode(t)
            ids = enc.ids
            if len(ids) <= 1:
                continue
            for i in range(1, len(ids)):
                start = max(0, i - context_size)
                context = ids[start:i]
                target = ids[i]
                self.examples.append((context, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def build_word_tokenizer(texts: List[str], vocab_size: int = 8000) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


# ===========================
# Model
# ===========================
class GatedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs, last_hidden):
        # encoder_outputs: (seq_len, batch, hidden)
        # last_hidden: (batch, hidden)
        keys = self.key(encoder_outputs)                 # (seq, batch, h)
        q = self.query(last_hidden).unsqueeze(0)         # (1, batch, h)
        scores = (keys * q).sum(dim=-1) / math.sqrt(keys.size(-1))  # (seq, batch)
        attn_w = F.softmax(scores, dim=0).unsqueeze(-1)  # (seq, batch, 1)
        context = (encoder_outputs * attn_w).sum(dim=0)  # (batch, h)
        gate_in = torch.cat([context, last_hidden], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in))         # (batch, h)
        combined = gate * context + (1.0 - gate) * last_hidden
        return F.relu(self.out(combined))                # (batch, h)


class PredictiveLSTM(nn.Module):
    """
    Word-level predictive model: embedding -> biLSTM -> gated-attn -> classifier
    """
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            batch_first=False, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gattn = GatedAttention(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor):
        # context_ids: (seq_len, batch)
        emb = self.emb(context_ids)                      # (seq, batch, emb)
        outputs, (hn, cn) = self.lstm(emb)               # outputs: (seq, batch, 2*hidden)
        outputs_proj = torch.tanh(self.proj(outputs))    # (seq, batch, hidden)
        last_fwd = hn[-2]                                # (batch, hidden)
        last_bwd = hn[-1]                                # (batch, hidden)
        last_hidden = torch.cat([last_fwd, last_bwd], dim=-1)  # (batch, 2*hidden)
        last_hidden = torch.tanh(self.proj(last_hidden))       # (batch, hidden)
        gated = self.gattn(outputs_proj, last_hidden)    # (batch, hidden)
        logits = self.output(gated)                      # (batch, vocab)
        return logits


# ===========================
# Training helpers
# ===========================
def collate_batch(batch):
    # batch: list of (context_tensor, target_tensor)
    contexts = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    maxlen = max(len(c) for c in contexts)
    padded = []
    for c in contexts:
        pad_len = maxlen - len(c)
        if pad_len > 0:
            padded.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), c]))
        else:
            padded.append(c)
    ctx = torch.stack(padded).transpose(0, 1)  # (seq_len, batch)
    return ctx, targets


def train_epoch(model, dataloader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for ctx, tgt in tqdm(dataloader, desc="train"):
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        optim.zero_grad()
        logits = model(ctx)                     # (batch, vocab)
        loss = criterion(logits, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for ctx, tgt in dataloader:
            ctx = ctx.to(device)
            tgt = tgt.to(device)
            logits = model(ctx)
            loss = criterion(logits, tgt)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ===========================
# Inference: top-k next words
# ===========================
def predict_top_k(model: PredictiveLSTM, tokenizer: Tokenizer, text_prefix: str, k: int = 5, device='cpu') -> List[Tuple[str, float]]:
    model.eval()
    ids = tokenizer.encode(text_prefix).ids
    if len(ids) == 0:
        return []
    # keep a short recent context (works fine for this small model)
    ctx = torch.tensor(ids[-(model.lstm.num_layers * 2 + 1):], dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(ctx)  # (1, vocab)
        probs = F.softmax(logits[0], dim=-1)
        topk = torch.topk(probs, k)
        out = []
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            tok = tokenizer.id_to_token(idx) if hasattr(tokenizer, "id_to_token") else str(idx)
            out.append((tok, float(p)))
    return out


# ===========================
# Train entrypoint
# ===========================
def example_train_run():
    # 1) Load corpus (try primary path, else fallback, else toy data)
    texts: List[str] = []
    src_used = None
    for path in [DATA_PATH, FALLBACK_PATH]:
        if path.exists():
            texts = load_corpus(path)
            src_used = str(path)
            break
    if not texts:
        # Toy backup so the script always runs
        texts = [
            "hello how are you",
            "hello world this is a test",
            "how are you doing today",
            "this is an example of predictive typing",
            "predictive typing helps speed up communication",
        ] * 200
        src_used = "<toy_data>"
    print(f"Loaded {len(texts)} lines from: {src_used}")

    # 2) Tokenizer + dataset
    tokenizer = build_word_tokenizer(texts, vocab_size=VOCAB_SIZE)
    vocab = tokenizer.get_vocab_size()
    ds = SimpleTextDataset(texts, tokenizer, context_size=CONTEXT_SIZE)

    # 3) Dataloader + model
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_idx = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else 0

    model = PredictiveLSTM(
        vocab_size=vocab,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        pad_idx=pad_idx
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4) Train
    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optim, criterion, device)
        print(f"Epoch {epoch+1} train loss {loss:.4f}")

    # 5) Save + quick inference test
    torch.save({
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_str()
    }, "predictive_lstm.pth")

    test_prefix = "the quick brown"
    preds = predict_top_k(model, tokenizer, test_prefix, k=5, device=device)
    print(f"Input: '{test_prefix}'")
    print("Preds:", preds)


if __name__ == "__main__":
    example_train_run()












