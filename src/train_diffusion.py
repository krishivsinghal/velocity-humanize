"""
Conditional DDPM on piece-centered velocity.

What diffuses: a scalar per note — velocity relative to mezzo-forte, scaled to
roughly unit variance (see VEL_SCALE).

What conditions: the deadpan note stream (pitch + timing), same as the
regression transformer. At diffusion time t we also feed the noisy velocity x_t
and a sinusoidal embedding of t into every token.

Training objective: predict the Gaussian noise epsilon in
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon.

At inference, start from pure noise and run the reverse chain to sample one
velocity curve; add back the piece mean if you want absolute MIDI levels.

  PYTHONPATH=src python src/train_diffusion.py
  PYTHONPATH=src python src/sample_diffusion.py
"""

from __future__ import annotations

from pathlib import Path
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusion_schedule import DDPMSchedule
from train_transformer import (
    BATCH_SIZE,
    CENTER_VEL,
    D_MODEL,
    DROPOUT,
    EMBED_LR,
    EncoderLayer,
    LR,
    MAX_LEN,
    N_HEADS,
    N_LAYERS,
    N_PITCH,
    N_TIMING,
    NUM_WORKERS,
    RotaryEmbedding,
    BalancedTrainBatches,
    PieceSet,
    VelocityDataset,
    collate,
    get_device,
)

PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
SAVE_PATH = PROCESSED / "velocity_diffusion.pt"

TIMESTEPS = 200
VEL_SCALE = 0.22  # centered 0–1 velocities usually sit within ~±0.25 of 64/127
EPOCHS = 8
STEPS_PER_EPOCH = 2500
# Full reverse chain matches training (TIMESTEPS). Subsampled DDPM stride is not valid;
# use sample_velocities_ddim for fewer steps.
SAMPLE_STEPS = TIMESTEPS


def encode_velocity(v: torch.Tensor) -> torch.Tensor:
    return (v - CENTER_VEL) / VEL_SCALE


def decode_velocity(x: torch.Tensor) -> torch.Tensor:
    return x * VEL_SCALE + CENTER_VEL


def sinusoidal_timestep(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10_000.0), half, device=t.device)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


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


class ConditionalVelocityDDPM(nn.Module):
    """Denoiser: deadpan notes + (x_t, t) -> predicted noise epsilon."""

    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING, D_MODEL)
        self.noisy_vel_proj = nn.Linear(1, D_MODEL)
        self.time_mlp = TimestepMLP()
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=MAX_LEN + 64)
        self.layers = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, 1)
        self.schedule = DDPMSchedule(TIMESTEPS)

    def forward(
        self,
        pitch: torch.Tensor,
        timing: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        t_vec = self.time_mlp(t).unsqueeze(1)
        x = (
            self.pitch_emb(pitch)
            + self.timing_proj(timing)
            + self.noisy_vel_proj(x_t.unsqueeze(-1))
            + t_vec
        )
        batch, seq = pad_mask.shape
        attn_mask = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, self.rope, attn_mask)
        eps = self.out(self.final_norm(x)).squeeze(-1)
        return eps


def batch_tensors(batch):
    pitch, timing, velocity, _win_id, _top, pad_mask = batch
    return pitch, timing, velocity, pad_mask


def diffusion_loss(model: ConditionalVelocityDDPM, batch, device):
    pitch, timing, velocity, pad_mask = [t.to(device) for t in batch_tensors(batch)]
    x0 = encode_velocity(velocity)
    b = x0.size(0)
    t = torch.randint(0, model.schedule.timesteps, (b,), device=device)
    noise = torch.randn_like(x0)
    x_t = model.schedule.q_sample(x0, t, noise)
    eps_pred = model(pitch, timing, x_t, t, pad_mask)
    loss = F.mse_loss(eps_pred[pad_mask], noise[pad_mask])
    return loss


@torch.no_grad()
def sample_velocities(
    model: ConditionalVelocityDDPM,
    pitch: torch.Tensor,
    timing: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    steps: int = SAMPLE_STEPS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full DDPM reverse chain (steps must equal schedule.timesteps).

    Striding with p_sample() is incorrect: each update assumes the previous index
    was t-1. For accelerated sampling use sample_velocities_ddim().
    """
    T = model.schedule.timesteps
    if steps != T:
        raise ValueError(
            f"sample_velocities expects steps={T} (full DDPM); got {steps}. "
            "Use sample_velocities_ddim() for fewer steps."
        )
    model.eval()
    device = pitch.device
    b, seq = pitch.shape
    x = torch.randn(b, seq, device=device, generator=generator)
    for t_val in range(T - 1, -1, -1):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        eps = model(pitch, timing, x, t, pad_mask)
        x = model.schedule.p_sample(x, t_val, eps, generator=generator)
    x = x.masked_fill(~pad_mask, 0.0)
    return decode_velocity(x)


@torch.no_grad()
def sample_velocities_ddim(
    model: ConditionalVelocityDDPM,
    pitch: torch.Tensor,
    timing: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    steps: int = 50,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DDIM reverse chain on a uniform subsequence of training timesteps."""
    T = model.schedule.timesteps
    if steps < 1 or steps > T:
        raise ValueError(f"steps must be in [1, {T}], got {steps}")
    model.eval()
    device = pitch.device
    b, seq = pitch.shape
    x = torch.randn(b, seq, device=device, generator=generator)
    time_idx = torch.linspace(T - 1, 0, steps + 1, device=device).long()
    for i in range(steps):
        t_val = int(time_idx[i].item())
        t_prev = int(time_idx[i + 1].item()) - 1 if time_idx[i + 1] > 0 else -1
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        eps = model(pitch, timing, x, t, pad_mask)
        x = model.schedule.ddim_step(
            x, t_val, t_prev, eps, eta=eta, generator=generator
        )
    x = x.masked_fill(~pad_mask, 0.0)
    return decode_velocity(x)


@torch.no_grad()
def eval_sample_rmse(model, loader, device, max_batches: int = 40):
    model.eval()
    se = 0.0
    n = 0
    pred_sq = pred_sum = true_sq = true_sum = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        pitch, timing, velocity, pad_mask = [t.to(device) for t in batch_tensors(batch)]
        pred = sample_velocities(model, pitch, timing, pad_mask, steps=SAMPLE_STEPS)
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
    print("device", device, "DDPM timesteps", TIMESTEPS, flush=True)
    print(
        f"balanced train: {STEPS_PER_EPOCH} steps/epoch, "
        f"{BATCH_SIZE // 2} ASAP + {BATCH_SIZE // 2} Giga",
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

    model = ConditionalVelocityDDPM().to(device)
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
        if val["rmse"] < best:
            best = val["rmse"]
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "timesteps": TIMESTEPS,
                    "vel_scale": VEL_SCALE,
                    "val_rmse": val["rmse"],
                },
                SAVE_PATH,
            )
            print("  saved", SAVE_PATH, flush=True)


if __name__ == "__main__":
    train()
