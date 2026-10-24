"""
Velocity diffusion with score-structure clues.

Same targets, split, batch, step budget, and 200-step sampler as
train_diffusion.py. The only change is the conditioning: measure position,
time signature, voice when the MusicXML lines up, and chord texture.
Those are clues. Nothing in the loss tells a downbeat or an octave to be louder.

  PYTHONPATH=src python src/train_diffusion_struct.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SpanDataset, VelocityDataset
from diffusion_schedule import DDPMSchedule
from features import N_TIMING_STRUCT, N_VOICE, assemble_timing, note_features
from score_structure import load_structure
from train_diffusion import (
    SAMPLE_STEPS,
    TIMESTEPS,
    VEL_SCALE,
    decode_velocity,
    encode_velocity,
    sinusoidal_timestep,
)
from train_transformer import (
    BATCH_SIZE,
    CENTER_VEL,
    CONTOUR_BEATS,
    D_MODEL,
    DROPOUT,
    EMBED_LR,
    EncoderLayer,
    LR,
    MAX_LEN,
    N_HEADS,
    N_LAYERS,
    N_PITCH,
    NUM_WORKERS,
    RotaryEmbedding,
    get_device,
)

PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
SAVE_PATH = PROCESSED / "velocity_diffusion_struct.pt"
LAST_PATH = PROCESSED / "velocity_diffusion_struct_last.pt"

EPOCHS = 8
STEPS_PER_EPOCH = 2500


class TimestepMLP(nn.Module):
    def __init__(self, dim: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_timestep(t, D_MODEL))


class StructuredVelocityDDPM(nn.Module):
    """Same denoiser as ConditionalVelocityDDPM, with a wider score input."""

    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.voice_emb = nn.Embedding(N_VOICE, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING_STRUCT, D_MODEL)
        self.noisy_vel_proj = nn.Linear(1, D_MODEL)
        self.time_mlp = TimestepMLP()
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=MAX_LEN + 64)
        self.layers = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, 1)
        self.schedule = DDPMSchedule(TIMESTEPS)

    def forward(self, pitch, timing, voice, x_t, t, pad_mask):
        t_vec = self.time_mlp(t).unsqueeze(1)
        x = (
            self.pitch_emb(pitch)
            + self.voice_emb(voice)
            + self.timing_proj(timing)
            + self.noisy_vel_proj(x_t.unsqueeze(-1))
            + t_vec
        )
        batch, seq = pad_mask.shape
        attn_mask = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, self.rope, attn_mask)
        return self.out(self.final_norm(x)).squeeze(-1)


class StructuredVelocityDataset(VelocityDataset):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        row = self.index.iloc[i]
        item["structure"] = load_structure(row["file"])
        item["file"] = row["file"]
        return item


def to_tensors(item, start, end):
    built = note_features(
        item["pitch"], item["start"], item["duration"], snap=bool(item["snap"])
    )
    timing, voice = assemble_timing(built, item.get("structure"))
    sl = slice(start, end)
    vel_raw = np.asarray(item["velocity"][sl], dtype=np.float32) / 127.0
    vel_centered = (
        vel_raw - float(vel_raw.mean()) + CENTER_VEL if len(vel_raw) > 0 else vel_raw
    )
    beat_pos = built["beat_pos"][sl]
    if len(beat_pos) > 0:
        win_id = np.floor((beat_pos - beat_pos[0]) / CONTOUR_BEATS).astype(np.int64)
        win_id = np.clip(win_id, 0, MAX_LEN - 1)
    else:
        win_id = np.zeros(0, dtype=np.int64)
    return {
        "pitch": torch.from_numpy(built["pitch"][sl].astype(np.int64)),
        "timing": torch.from_numpy(np.ascontiguousarray(timing[sl])),
        "voice": torch.from_numpy(np.ascontiguousarray(voice[sl])),
        "velocity": torch.from_numpy(vel_centered.astype(np.float32)),
        "win_id": torch.from_numpy(win_id),
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


class BalancedTrainBatches(torch.utils.data.IterableDataset):
    def __init__(self, steps_per_epoch, crop):
        super().__init__()
        self.steps = steps_per_epoch
        self.crop = crop
        self.asap = StructuredVelocityDataset(split="train")
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
    count = len(batch)
    pitch = torch.zeros(count, max_len, dtype=torch.long)
    voice = torch.zeros(count, max_len, dtype=torch.long)
    timing = torch.zeros(count, max_len, N_TIMING_STRUCT)
    velocity = torch.zeros(count, max_len)
    pad_mask = torch.zeros(count, max_len, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x["pitch"].size(0)
        pitch[i, :n] = x["pitch"]
        voice[i, :n] = x["voice"]
        timing[i, :n] = x["timing"]
        velocity[i, :n] = x["velocity"]
        pad_mask[i, :n] = True
    return pitch, timing, voice, velocity, pad_mask


def diffusion_loss(model, batch, device):
    pitch, timing, voice, velocity, pad_mask = [t.to(device) for t in batch]
    x0 = encode_velocity(velocity)
    t = torch.randint(0, model.schedule.timesteps, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    x_t = model.schedule.q_sample(x0, t, noise)
    eps_pred = model(pitch, timing, voice, x_t, t, pad_mask)
    return F.mse_loss(eps_pred[pad_mask], noise[pad_mask])


@torch.no_grad()
def sample_velocities(model, pitch, timing, voice, pad_mask, *, steps=SAMPLE_STEPS, generator=None):
    """Full DDPM reverse chain. steps must equal the training schedule length."""
    T = model.schedule.timesteps
    if steps != T:
        raise ValueError(
            f"sample_velocities expects steps={T} (full DDPM); got {steps}."
        )
    model.eval()
    device = pitch.device
    b, seq = pitch.shape
    x = torch.randn(b, seq, device=device, generator=generator)
    for t_val in range(T - 1, -1, -1):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        eps = model(pitch, timing, voice, x, t, pad_mask)
        x = model.schedule.p_sample(x, t_val, eps, generator=generator)
    x = x.masked_fill(~pad_mask, 0.0)
    return decode_velocity(x)


@torch.no_grad()
def eval_sample_rmse(model, loader, device, max_batches: int = 40):
    model.eval()
    se = n = 0.0
    pred_sq = pred_sum = true_sq = true_sum = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        pitch, timing, voice, velocity, pad_mask = [t.to(device) for t in batch]
        pred = sample_velocities(model, pitch, timing, voice, pad_mask, steps=SAMPLE_STEPS)
        p = pred[pad_mask]
        y = velocity[pad_mask]
        se += (p - y).pow(2).sum().item()
        pred_sum += p.sum().item()
        pred_sq += p.pow(2).sum().item()
        true_sum += y.sum().item()
        true_sq += y.pow(2).sum().item()
        n += y.numel()
    pred_var = (pred_sq / n) - (pred_sum / n) ** 2
    true_var = (true_sq / n) - (true_sum / n) ** 2
    return {
        "rmse": (se / n) ** 0.5 * 127,
        "pred_std": max(pred_var, 0.0) ** 0.5 * 127,
        "true_std": max(true_var, 0.0) ** 0.5 * 127,
    }


def train():
    random.seed(1)
    torch.manual_seed(1)
    device = get_device()
    print("device", device, "DDPM timesteps", TIMESTEPS, "timing", N_TIMING_STRUCT, flush=True)
    print(
        f"balanced train: {STEPS_PER_EPOCH} steps/epoch, "
        f"{BATCH_SIZE // 2} ASAP + {BATCH_SIZE // 2} Giga",
        flush=True,
    )

    train_loader = BalancedTrainBatches(STEPS_PER_EPOCH, crop=True)
    val_ds = torch.utils.data.ConcatDataset(
        [
            _IndexDataset(StructuredVelocityDataset(split="val"), crop=False),
            _IndexDataset(SpanDataset(split="val"), crop=False),
        ]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate,
    )

    model = StructuredVelocityDDPM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params}", flush=True)
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in model.named_parameters() if name.endswith("emb.weight")],
                "lr": EMBED_LR,
            },
            {
                "params": [p for name, p in model.named_parameters() if not name.endswith("emb.weight")],
                "lr": LR,
            },
        ]
    )

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            opt.zero_grad()
            loss = diffusion_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            seen += 1
            if seen == 1 or seen % 500 == 0:
                print(
                    f"  epoch {epoch} step {seen}/{STEPS_PER_EPOCH}  "
                    f"eps_mse {running / seen:.4f}",
                    flush=True,
                )
        val = eval_sample_rmse(model, val_loader, device)
        print(
            f"epoch {epoch}  train eps_mse {running / seen:.4f}  "
            f"sample val RMSE {val['rmse']:.2f}  "
            f"pred std {val['pred_std']:.1f}  true std {val['true_std']:.1f}",
            flush=True,
        )
        payload = {
            "model": model.state_dict(),
            "timesteps": TIMESTEPS,
            "vel_scale": VEL_SCALE,
            "val_rmse": val["rmse"],
            "epoch": epoch,
            "n_timing": N_TIMING_STRUCT,
            "n_voice": N_VOICE,
        }
        LAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, LAST_PATH)
        if val["rmse"] < best:
            best = val["rmse"]
            torch.save(payload, SAVE_PATH)
            print("  saved", SAVE_PATH, flush=True)


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, inner, crop):
        self.inner = inner
        self.crop = crop

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return crop_tensors(self.inner, i, self.crop)


if __name__ == "__main__":
    train()
