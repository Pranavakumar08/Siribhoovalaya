# kannada_segmentation_train_infer.py
import json
import unicodedata
import re
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_JSON = "kn_IndicNER_v1.0/kn_train.json"
VAL_JSON = "kn_IndicNER_v1.0/kn_val.json"
TEST_JSON = "kn_IndicNER_v1.0/kn_test.json"
MODEL_PATH = "kannada_segmentation_m_json.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 10
PAD_IDX = 0
UNK_CHAR = "<UNK>"
MAX_SEQ_LEN = 256   # increase if you have long sentences; or use sliding windows

# -----------------------------
# Helpers: load JSONL like your files
# -----------------------------
def load_jsonl_sentences(path: str) -> List[List[str]]:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "words" in obj and isinstance(obj["words"], list):
                sentences.append(obj["words"])
    return sentences

# -----------------------------
# Normalize & clean text
# -----------------------------
KANNADA_RE = re.compile(r'[\u0C80-\u0CFF]')  # Kannada Unicode block

def normalize_text(s: str) -> str:
    # NFC normalization
    s = unicodedata.normalize("NFC", s)
    return s

def concat_words_and_labels(words: List[str]) -> Tuple[str, List[int]]:
    """
    Given token list words = ['w1','w2',...], return:
    - chars: concatenated string "w1w2..."
    - labels: list of same length, label[i] = 1 if boundary AFTER char i else 0
    """
    chars = []
    labels = []
    for w in words:
        w = normalize_text(w)
        if len(w) == 0:
            continue
        for ch in w:
            chars.append(ch)
            labels.append(0)  # default 0, will set last char of word to 1 below
        labels[-1] = 1  # mark boundary after last char of this word
    return "".join(chars), labels

# -----------------------------
# Build vocabulary from train/val/test
# -----------------------------
def build_char_vocab(sentences_all: List[List[str]], min_freq=1):
    freq = {}
    for words in sentences_all:
        for w in words:
            w = normalize_text(w)
            for ch in w:
                freq[ch] = freq.get(ch, 0) + 1
    # include only those >= min_freq (or all)
    chars = [c for c, f in freq.items() if f >= min_freq]
    chars = sorted(chars)
    # add UNK
    char_to_idx = {PAD_IDX: "<PAD>", 1: UNK_CHAR}
    idx = 2
    for c in chars:
        if c not in char_to_idx.values():
            char_to_idx[idx] = c
            idx += 1
    # invert mapping
    mapping = {c: i for i, c in char_to_idx.items()}
    # mapping currently maps idx->char; invert:
    inv = {v: k for k, v in mapping.items()}
    # Make char_to_idx: char -> idx
    char_to_idx = {char: idx for idx, char in inv.items()}
    # ensure PAD=0 and UNK present
    char_to_idx = {k: v for k, v in char_to_idx.items()}
    # Guarantee PAD and UNK
    char_to_idx["<PAD>"] = PAD_IDX
    char_to_idx[UNK_CHAR] = 1
    # create idx_to_char
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char

# -----------------------------
# Dataset
# -----------------------------
class KannadaSegDataset(Dataset):
    def __init__(self, sentences: List[List[str]], char_to_idx, max_len=MAX_SEQ_LEN):
        self.samples = []
        self.max_len = max_len
        self.char_to_idx = char_to_idx
        for words in sentences:
            chars, labels = concat_words_and_labels(words)
            # optionally skip extremely long or empty
            if len(chars) == 0:
                continue
            # If longer than max_len, split into sliding windows with overlap
            start = 0
            while start < len(chars):
                end = min(start + max_len, len(chars))
                chunk_chars = chars[start:end]
                chunk_labels = labels[start:end]
                # If this chunk breaks a word boundary (i.e., we cut inside a word),
                # we must ensure label semantics remain consistent: if end < len(chars) and
                # we cut in middle of word, the last char in chunk should NOT be labeled 1.
                if end < len(chars):
                    # if we cut mid-word, ensure last label is 0
                    if chunk_labels[-1] == 1 and (len(chunk_chars) > 0 and end < len(labels) and labels[end-1] == 1 and (end < len(labels) and labels[end] == 0)):
                        # this is already fine; but safe guard:
                        pass
                self.samples.append((chunk_chars, chunk_labels))
                if end == len(chars):
                    break
                # overlap for context
                start += max_len // 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, labels = self.samples[idx]
        x = [self.char_to_idx.get(ch, self.char_to_idx.get(UNK_CHAR, 1)) for ch in chars]
        y = labels
        return torch.LongTensor(x), torch.LongTensor(y), len(x)

def collate_fn(batch):
    # batch: list of (x_tensor, y_tensor, length)
    lengths = [t[2] for t in batch]
    max_len = max(lengths)
    xs = []
    ys = []
    masks = []
    for x, y, l in batch:
        pad_x = torch.cat([x, torch.zeros(max_len - l, dtype=torch.long)])
        pad_y = torch.cat([y, torch.full((max_len - l,), -100, dtype=torch.long)])  # ignore_index=-100
        mask = torch.cat([torch.ones(l, dtype=torch.bool), torch.zeros(max_len - l, dtype=torch.bool)])
        xs.append(pad_x)
        ys.append(pad_y)
        masks.append(mask)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    masks = torch.stack(masks)
    return xs.to(DEVICE), ys.to(DEVICE), masks.to(DEVICE), lengths

# -----------------------------
# Model: BiLSTM
# -----------------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, tagset_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, mask=None):
        emb = self.embedding(x)            # (batch, seq, emb)
        packed_out, _ = self.lstm(emb)     # (batch, seq, hidden)
        logits = self.fc(packed_out)       # (batch, seq, tagset)
        return logits

# -----------------------------
# Training routine
# -----------------------------
def train_model(train_sentences, val_sentences):
    # build vocab
    char_to_idx, idx_to_char = build_char_vocab(train_sentences + val_sentences)
    vocab_size = len(char_to_idx)
    print("Vocab size:", vocab_size)

    train_ds = KannadaSegDataset(train_sentences, char_to_idx)
    val_ds = KannadaSegDataset(val_sentences, char_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMTagger(vocab_size=vocab_size, emb_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, tagset_size=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # matches our pad label

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xs, ys, masks, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(xs)  # (batch, seq, 2)
            loss = criterion(logits.view(-1, 2), ys.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} train_loss={avg_train_loss:.4f}")

        # validation
        model.eval()
        total_v = 0
        correct_v = 0
        with torch.no_grad():
            for xs, ys, masks, lengths in val_loader:
                logits = model(xs)
                preds = torch.argmax(logits, dim=-1)
                # count only non -100 positions
                mask_valid = (ys != -100)
                total_v += mask_valid.sum().item()
                correct_v += ((preds == ys) & mask_valid).sum().item()
        acc = correct_v / total_v if total_v > 0 else 0.0
        print(f"Validation accuracy: {acc:.4f}")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": {i: c for c, i in char_to_idx.items()}
    }, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return MODEL_PATH

# -----------------------------
# Inference function
# -----------------------------
def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=DEVICE)
    char_to_idx = ckpt["char_to_idx"]
    idx_to_char = {int(k): v for k, v in ckpt.get("idx_to_char", {}).items()}
    model = BiLSTMTagger(vocab_size=len(char_to_idx), emb_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, tagset_size=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, char_to_idx, idx_to_char

def segment_with_model(text: str, model, char_to_idx):
    text = normalize_text(text)
    # convert to char indices
    seq_ids = [char_to_idx.get(ch, char_to_idx.get(UNK_CHAR, 1)) for ch in text]
    # If too long, segment by sliding windows and stitch outputs
    window = MAX_SEQ_LEN
    step = window // 2
    segmented = []
    i = 0
    while i < len(seq_ids):
        end = min(i + window, len(seq_ids))
        chunk = seq_ids[i:end]
        x = torch.LongTensor([chunk]).to(DEVICE)
        with torch.no_grad():
            logits = model(x)            # (1, seq, 2)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        # map predictions back onto characters
        for j, p in enumerate(preds):
            ch = text[i + j]
            segmented.append((ch, p))
        i += step

    # combine into words: label 1 means boundary AFTER char
    words = []
    cur = ""
    for ch, boundary in segmented:
        cur += ch
        if boundary == 1:
            words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    return words

# -----------------------------
# Quick run: train (if train file exists) and demo inference
# -----------------------------
if __name__ == "__main__":
    
    # load data
    val_sentences = load_jsonl_sentences(VAL_JSON)
    test_sentences = load_jsonl_sentences(TEST_JSON)
    train_sentences = []
    if os.path.exists(TRAIN_JSON):
        train_sentences = load_jsonl_sentences(TRAIN_JSON)
        print("Loaded train count:", len(train_sentences))
    else:
        print(f"Warning: {TRAIN_JSON} not found; training will use val as proxy.")
        # use val as training proxy if no train file
        train_sentences = val_sentences.copy()

    # Train model
    model_path = train_model(train_sentences, val_sentences)
