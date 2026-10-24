"""
Deadpan notes -> piece-centered velocity (stage 1).

Encoder-only transformer with RoPE; Gaussian NLL in MIDI units on velocity
(mean + log-variance per note) plus 2× contour loss on 4-beat windows.
Training batches are half ASAP, half GigaMIDI. Onset offset is deferred.

  source venv_ml/bin/activate
  PYTHONPATH=src python src/train_transformer.py
"""

from __future__ import annotations

from pathlib import Path
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dataset import SpanDataset, VelocityDataset, all_notes
from features import PITCH_CLASS_NAMES, note_features

PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
SAVE_PATH = PROCESSED / "velocity_transformer.pt"

D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
SWIGLU_HIDDEN = 384
MAX_LEN = 1024
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 10
STEPS_PER_EPOCH = 5000
LR = 3e-4
EMBED_LR = 1e-4
DROPOUT = 0.1
N_PITCH = 128
N_TIMING = 4
CENTER_VEL = 64.0 / 127.0
CONTOUR_BEATS = 4.0
CONTOUR_WEIGHT = 2.0
# Huber on centered 0–1 velocities for contour (~10 MIDI).
HUBER_DELTA = 0.08
# Initial log(variance in MIDI^2) ≈ (15 velocity levels)^2.
INIT_LOG_VAR_MIDI = math.log(15.0**2)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def note_name(pitch):
    return f"{PITCH_CLASS_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        seq = x.size(2)
        cos = self.cos[:, :, :seq, :]
        sin = self.sin[:, :, :seq, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, SWIGLU_HIDDEN)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope, attn_mask):
        h = self.norm1(x)
        batch, seq, _ = h.shape
        qkv = self.qkv(h).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = rope(q.transpose(1, 2))
        k = rope(k.transpose(1, 2))
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        x = x + self.drop(self.proj(attn))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VelocityTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING, D_MODEL)
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=MAX_LEN + 64)
        self.layers = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = INIT_LOG_VAR_MIDI

    def forward(self, pitch, timing, pad_mask):
        x = self.pitch_emb(pitch) + self.timing_proj(timing)
        batch, seq = pad_mask.shape
        attn_mask = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, self.rope, attn_mask)
        raw = self.head(self.final_norm(x))
        mu = CENTER_VEL + raw[..., 0]
        log_var_midi = raw[..., 1].clamp(-2.0, 10.0)
        return mu, log_var_midi


def to_tensors(item, start, end):
    built = note_features(
        item["pitch"], item["start"], item["duration"], snap=bool(item["snap"])
    )
    sl = slice(start, end)
    vel_raw = np.asarray(item["velocity"][sl], dtype=np.float32) / 127.0
    vel_centered = (
        vel_raw - float(vel_raw.mean()) + CENTER_VEL if len(vel_raw) > 0 else vel_raw
    )
    timing = built["timing"][sl]
    beat_pos = built["beat_pos"][sl]
    height = timing[:, 3]
    is_top = height >= 0.49
    if len(beat_pos) > 0:
        win_id = np.floor((beat_pos - beat_pos[0]) / CONTOUR_BEATS).astype(np.int64)
        win_id = np.clip(win_id, 0, MAX_LEN - 1)
    else:
        win_id = np.zeros(0, dtype=np.int64)
    return {
        "pitch": torch.from_numpy(built["pitch"][sl].astype(np.int64)),
        "timing": torch.from_numpy(timing.astype(np.float32)),
        "velocity": torch.from_numpy(vel_centered.astype(np.float32)),
        "win_id": torch.from_numpy(win_id),
        "is_top": torch.from_numpy(is_top),
    }


def crop_tensors(ds, index, crop):
    item = ds[index]
    n = len(item["pitch"])
    if crop and n > MAX_LEN:
        s = random.randint(0, n - MAX_LEN)
        e = s + MAX_LEN
    else:
        s, e = 0, min(n, MAX_LEN)
    return to_tensors(item, s, e)


class PieceSet(Dataset):
    def __init__(self, split, crop):
        self.inner = all_notes(split)
        self.crop = crop

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return crop_tensors(self.inner, i, self.crop)


class BalancedTrainBatches(IterableDataset):
    """Each batch: BATCH_SIZE//2 ASAP + BATCH_SIZE//2 GigaMIDI (random with replacement)."""

    def __init__(self, steps_per_epoch, crop):
        super().__init__()
        self.steps = steps_per_epoch
        self.crop = crop
        self.asap = VelocityDataset(split="train")
        self.giga = SpanDataset(split="train")

    def __len__(self):
        return self.steps

    def __iter__(self):
        half = BATCH_SIZE // 2
        n_asap = len(self.asap)
        n_giga = len(self.giga)
        for _ in range(self.steps):
            batch = []
            for _ in range(half):
                batch.append(crop_tensors(self.asap, random.randrange(n_asap), self.crop))
            for _ in range(half):
                batch.append(crop_tensors(self.giga, random.randrange(n_giga), self.crop))
            random.shuffle(batch)
            yield collate(batch)


def collate(batch):
    max_len = max(x["pitch"].size(0) for x in batch)
    B = len(batch)
    pitch = torch.zeros(B, max_len, dtype=torch.long)
    timing = torch.zeros(B, max_len, N_TIMING)
    velocity = torch.zeros(B, max_len)
    win_id = torch.zeros(B, max_len, dtype=torch.long)
    top_mask = torch.zeros(B, max_len, dtype=torch.bool)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x["pitch"].size(0)
        pitch[i, :n] = x["pitch"]
        timing[i, :n] = x["timing"]
        velocity[i, :n] = x["velocity"]
        win_id[i, :n] = x["win_id"]
        top_mask[i, :n] = x["is_top"]
        pad_mask[i, :n] = True
    return pitch, timing, velocity, win_id, top_mask, pad_mask


def velocity_nll(mu, log_var_midi, target, mask):
    """Gaussian NLL in MIDI velocity units (readable scale, rewards matching spread)."""
    mu_m = mu * 127.0
    y_m = target * 127.0
    nll = 0.5 * (((y_m - mu_m) ** 2) / log_var_midi.exp() + log_var_midi)
    return nll[mask].mean()


def contour_loss(pred, target, win_id, top_mask):
    B, S = pred.shape
    n_wins = int(MAX_LEN)
    m = top_mask.to(pred.dtype)
    counts = torch.zeros(B, n_wins, device=pred.device, dtype=pred.dtype)
    sum_p = torch.zeros(B, n_wins, device=pred.device, dtype=pred.dtype)
    sum_t = torch.zeros(B, n_wins, device=pred.device, dtype=pred.dtype)
    counts.scatter_add_(1, win_id, m)
    sum_p.scatter_add_(1, win_id, pred * m)
    sum_t.scatter_add_(1, win_id, target * m)
    denom = counts.gather(1, win_id).clamp_min(1.0)
    mean_p = sum_p.gather(1, win_id) / denom
    mean_t = sum_t.gather(1, win_id) / denom
    valid = top_mask & (counts.gather(1, win_id) >= 2.0)
    if not bool(valid.any()):
        return pred.new_zeros(())
    return F.huber_loss((pred - mean_p)[valid], (target - mean_t)[valid], delta=HUBER_DELTA)


def batch_loss(model, batch, device):
    pitch, timing, velocity, win_id, top_mask, pad_mask = [
        t.to(device, non_blocking=True) for t in batch
    ]
    mu, log_var = model(pitch, timing, pad_mask)
    vel = velocity_nll(mu, log_var, velocity, pad_mask)
    contour = contour_loss(mu, velocity, win_id, top_mask)
    total = vel + CONTOUR_WEIGHT * contour
    parts = {"vel": float(vel.detach()), "contour": float(contour.detach())}
    return total, parts


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    vel_se = vel_ae = 0.0
    n = 0
    pred_sum = pred_sq = 0.0
    true_sum = true_sq = 0.0
    for batch in loader:
        pitch, timing, velocity, win_id, top_mask, pad_mask = [t.to(device) for t in batch]
        mu, _ = model(pitch, timing, pad_mask)
        p = mu[pad_mask]
        y = velocity[pad_mask]
        vel_se += (p - y).pow(2).sum().item()
        vel_ae += (p - y).abs().sum().item()
        pred_sum += p.sum().item()
        pred_sq += p.pow(2).sum().item()
        true_sum += y.sum().item()
        true_sq += y.pow(2).sum().item()
        n += y.numel()
    mean_y = true_sum / n
    baseline = (true_sq / n) - mean_y**2
    pred_var = (pred_sq / n) - (pred_sum / n) ** 2
    return {
        "rmse": (vel_se / n) ** 0.5 * 127,
        "mae": (vel_ae / n) * 127,
        "pred_std": max(pred_var, 0.0) ** 0.5 * 127,
        "true_std": baseline**0.5 * 127,
        "baseline_rmse": baseline**0.5 * 127,
    }


@torch.no_grad()
def show_example(model, device):
    model.eval()
    item = VelocityDataset(split="train")[0]
    tensors = to_tensors(item, 0, min(16, len(item["pitch"])))
    pad = torch.ones(1, tensors["pitch"].size(0), dtype=torch.bool, device=device)
    mu, _ = model(
        tensors["pitch"].unsqueeze(0).to(device),
        tensors["timing"].unsqueeze(0).to(device),
        pad,
    )
    true_v = tensors["velocity"] * 127
    pred_v = mu[0].cpu() * 127
    print(item["composer"], item["title"], flush=True)
    print("note   trueV predV", flush=True)
    for i in range(len(true_v)):
        print(
            f"{note_name(tensors['pitch'][i]):4s}  {true_v[i]:5.1f} {pred_v[i]:5.1f}",
            flush=True,
        )
    print(
        f"sample true vel std {true_v.std():.1f}  pred vel std {pred_v.std():.1f}",
        flush=True,
    )


def train():
    random.seed(0)
    torch.manual_seed(0)
    device = get_device()
    print("device", device, flush=True)
    print(
        f"balanced train: {STEPS_PER_EPOCH} steps/epoch, "
        f"{BATCH_SIZE // 2} ASAP + {BATCH_SIZE // 2} Giga per batch",
        flush=True,
    )

    train_loader = BalancedTrainBatches(STEPS_PER_EPOCH, crop=True)
    val_loader = DataLoader(
        PieceSet("val", crop=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=collate,
    )

    model = VelocityTransformer().to(device)
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if n.endswith("emb.weight")],
                "lr": EMBED_LR,
            },
            {
                "params": [p for n, p in model.named_parameters() if not n.endswith("emb.weight")],
                "lr": LR,
            },
        ]
    )

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        running_parts = {"vel": 0.0, "contour": 0.0}
        seen = 0
        for batch in train_loader:
            opt.zero_grad()
            loss, parts = batch_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            for key in running_parts:
                running_parts[key] += parts[key]
            seen += 1
            if seen == 1 or seen % 500 == 0:
                print(
                    f"  epoch {epoch} step {seen}/{STEPS_PER_EPOCH}  "
                    f"loss {running / seen:.3f}  "
                    f"nll {running_parts['vel'] / seen:.3f}  "
                    f"contour {running_parts['contour'] / seen:.4f}",
                    flush=True,
                )
        val = eval_metrics(model, val_loader, device)
        print(
            f"epoch {epoch}  train loss {running / seen:.3f}  "
            f"val RMSE {val['rmse']:.2f}  MAE {val['mae']:.2f}  "
            f"pred std {val['pred_std']:.1f}  true std {val['true_std']:.1f}  "
            f"mean-baseline RMSE {val['baseline_rmse']:.2f}",
            flush=True,
        )
        if val["rmse"] < best:
            best = val["rmse"]
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_rmse": val["rmse"]}, SAVE_PATH)
            print("  saved", SAVE_PATH, flush=True)

    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print("\n--- sample ---", flush=True)
    show_example(model, device)


if __name__ == "__main__":
    train()
