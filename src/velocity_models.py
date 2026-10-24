"""Velocity-only diffusion and score-encoder / performance-decoder.

Widths match the existing blocks: d_model 96, 4 heads, SwiGLU 384, dropout 0.1.
Diffusion uses four bidirectional blocks. The decoder uses four score-encoder
blocks and two causal performance-decoder blocks. Positional encoding is the
existing rotary embedding. No pitch, timing, duration, or pedal head.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_schedule import DDPMSchedule
from features import N_TIMING_STRUCT, N_VOICE
from train_diffusion import TimestepMLP
from train_seq2seq import DecoderLayer
from train_transformer import DROPOUT, D_MODEL, EncoderLayer, N_HEADS, N_PITCH, RotaryEmbedding

TIMESTEPS = 200
CONTEXT = 256
N_VELOCITY = 127  # MIDI velocities 1..127


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


class VelocityDiffusion(nn.Module):
    """Epsilon denoiser conditioned on the score, noisy velocity, and timestep."""

    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.voice_emb = nn.Embedding(N_VOICE, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING_STRUCT, D_MODEL)
        # x_t plus an availability flag. Missing candidates are not a velocity.
        self.noisy_proj = nn.Linear(2, D_MODEL)
        self.time_mlp = TimestepMLP(D_MODEL)
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=CONTEXT + 64)
        self.layers = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(4)]
        )
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, 1)
        self.schedule = DDPMSchedule(TIMESTEPS)

    def forward(self, pitch, timing, voice, x_t, available, t, pad_mask):
        avail = available.to(dtype=x_t.dtype).unsqueeze(-1)
        noisy = self.noisy_proj(torch.cat([x_t.unsqueeze(-1), avail], dim=-1))
        x = (
            self.pitch_emb(pitch)
            + self.voice_emb(voice)
            + self.timing_proj(timing)
            + noisy
            + self.time_mlp(t).unsqueeze(1)
        )
        batch, seq = pad_mask.shape
        attn = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        attn = attn.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, self.rope, attn)
        return self.out(self.final_norm(x)).squeeze(-1)


class VelocityScoreDecoder(nn.Module):
    """Bidirectional score encoder, causal velocity decoder, 127-way head."""

    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.voice_emb = nn.Embedding(N_VOICE, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING_STRUCT, D_MODEL)
        # (v-64)/32, BOS indicator, unknown-history indicator.
        self.hist_proj = nn.Linear(3, D_MODEL)
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=CONTEXT + 64)
        self.encoder = nn.ModuleList(
            [EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(4)]
        )
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(2)])
        self.enc_norm = nn.LayerNorm(D_MODEL)
        self.dec_norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_VELOCITY)

    def encode(self, pitch, timing, voice, pad_mask):
        x = self.pitch_emb(pitch) + self.voice_emb(voice) + self.timing_proj(timing)
        batch, seq = pad_mask.shape
        attn = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        attn = attn.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.encoder:
            x = layer(x, self.rope, attn)
        return self.enc_norm(x)

    def history(self, velocity, available):
        """Shifted previous velocity. Position 0 is BOS. Missing history is unknown."""
        batch, seq = velocity.shape
        scaled = torch.zeros(batch, seq, device=velocity.device, dtype=velocity.dtype)
        bos = torch.zeros(batch, seq, device=velocity.device, dtype=velocity.dtype)
        unknown = torch.zeros(batch, seq, device=velocity.device, dtype=velocity.dtype)
        bos[:, 0] = 1.0
        if seq > 1:
            prev_ok = available[:, :-1]
            prev = velocity[:, :-1]
            scaled[:, 1:] = torch.where(prev_ok, (prev - 64.0) / 32.0, torch.zeros_like(prev))
            unknown[:, 1:] = (~prev_ok).to(dtype=velocity.dtype)
        return self.hist_proj(torch.stack([scaled, bos, unknown], dim=-1))

    def decode(self, memory, velocity, available, pad_mask):
        dec = self.history(velocity, available) + memory
        batch, seq = pad_mask.shape
        causal = torch.triu(torch.full((seq, seq), float("-inf"), device=dec.device), diagonal=1)
        pad = torch.zeros(batch, 1, 1, seq, device=dec.device, dtype=dec.dtype)
        pad = pad.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        self_mask = causal.view(1, 1, seq, seq) + pad
        for layer in self.decoder:
            dec = layer(dec, memory, self_mask, pad, self.rope)
        return self.head(self.dec_norm(dec))

    def forward(self, pitch, timing, voice, velocity, available, pad_mask):
        memory = self.encode(pitch, timing, voice, pad_mask)
        return self.decode(memory, velocity, available, pad_mask)


def diffusion_loss(model, batch, mu, sigma):
    pitch = batch["pitch"]
    timing = batch["timing"]
    voice = batch["voice"]
    velocity = batch["velocity"]
    available = batch["available"]
    pad_mask = batch["pad_mask"]
    mask = pad_mask & available
    if int(mask.sum()) == 0:
        return None
    x0 = torch.zeros_like(velocity)
    x0 = torch.where(mask, (velocity - mu) / sigma, x0)
    b = x0.size(0)
    # CPU RNG so a saved torch generator state resumes the same noise.
    t = torch.randint(0, model.schedule.timesteps, (b,))
    noise = torch.randn(x0.shape)
    t = t.to(x0.device)
    noise = noise.to(x0.device)
    x_t = model.schedule.q_sample(x0, t, noise)
    x_t = torch.where(mask, x_t, torch.zeros_like(x_t))
    eps = model(pitch, timing, voice, x_t, mask, t, pad_mask)
    return F.mse_loss(eps[mask], noise[mask])


def decoder_loss(model, batch):
    logits = model(
        batch["pitch"],
        batch["timing"],
        batch["voice"],
        batch["velocity"],
        batch["available"],
        batch["pad_mask"],
    )
    mask = batch["pad_mask"] & batch["available"]
    if int(mask.sum()) == 0:
        return None
    target = (batch["velocity"][mask].long() - 1).clamp(0, N_VELOCITY - 1)
    return F.cross_entropy(logits[mask], target)


@torch.no_grad()
def sample_diffusion(model, pitch, timing, voice, pad_mask, mu, sigma, generator=None):
    """Full 200-step DDPM. Every real note is a candidate. Returns raw MIDI velocity."""
    steps = model.schedule.timesteps
    if steps != TIMESTEPS:
        raise ValueError(f"expected T={TIMESTEPS}, schedule has {steps}")
    model.eval()
    device = pitch.device
    b, seq = pitch.shape
    x = torch.randn(b, seq, device=device, generator=generator)
    available = pad_mask
    for t_val in range(steps - 1, -1, -1):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        eps = model(pitch, timing, voice, x, available, t, pad_mask)
        x = model.schedule.p_sample(x, t_val, eps, generator=generator)
        x = torch.where(pad_mask, x, torch.zeros_like(x))
    midi = x * sigma + mu
    return midi.masked_fill(~pad_mask, 0.0)


@torch.no_grad()
def sample_diffusion_ddim(model, pitch, timing, voice, pad_mask, mu, sigma, steps=50, eta=0.0, generator=None):
    """Optional 50-step DDIM. Not used for official evaluation."""
    T = model.schedule.timesteps
    if steps < 1 or steps > T:
        raise ValueError(f"steps must be in [1, {T}], got {steps}")
    model.eval()
    device = pitch.device
    b, seq = pitch.shape
    x = torch.randn(b, seq, device=device, generator=generator)
    available = pad_mask
    time_idx = torch.linspace(T - 1, 0, steps + 1, device=device).long()
    for i in range(steps):
        t_val = int(time_idx[i].item())
        t_prev = int(time_idx[i + 1].item()) - 1 if time_idx[i + 1] > 0 else -1
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        eps = model(pitch, timing, voice, x, available, t, pad_mask)
        x = model.schedule.ddim_step(x, t_val, t_prev, eps, eta=eta, generator=generator)
        x = torch.where(pad_mask, x, torch.zeros_like(x))
    return (x * sigma + mu).masked_fill(~pad_mask, 0.0)


@torch.no_grad()
def rollout_decoder(model, pitch, timing, voice, pad_mask, mode="greedy", temperature=1.0, generator=None):
    """Free-running decoder. History is its own previous outputs, BOS at the start."""
    model.eval()
    batch, seq = pitch.shape
    velocity = torch.zeros(batch, seq, device=pitch.device)
    available = torch.zeros(batch, seq, dtype=torch.bool, device=pitch.device)
    chosen = torch.zeros(batch, seq, device=pitch.device)
    memory = model.encode(pitch, timing, voice, pad_mask)
    for i in range(seq):
        logits = model.decode(memory, velocity, available, pad_mask)
        step = logits[:, i, :]
        if mode == "greedy":
            idx = step.argmax(dim=-1)
        elif mode == "sample":
            probs = torch.softmax(step / temperature, dim=-1)
            # CPU draw so the seed does not depend on the MPS generator.
            idx = torch.multinomial(probs.detach().float().cpu(), 1, generator=generator)
            idx = idx.squeeze(-1).to(step.device)
        else:
            raise ValueError(mode)
        vel = (idx + 1).to(dtype=velocity.dtype)
        real = pad_mask[:, i]
        chosen[:, i] = torch.where(real, vel, torch.zeros_like(vel))
        velocity[:, i] = chosen[:, i]
        available[:, i] = real
    return chosen


def expected_velocity(logits):
    """Diagnostic mean of the categorical distribution. Not a performance output."""
    probs = torch.softmax(logits, dim=-1)
    values = torch.arange(1, N_VELOCITY + 1, device=logits.device, dtype=probs.dtype)
    return (probs * values).sum(dim=-1)


def masked_epsilon_mse(model, batch, mu, sigma, timesteps, seed):
    """Panel score: masked noise MSE at fixed timesteps and a fixed noise draw."""
    model.eval()
    velocity = batch["velocity"]
    mask = batch["pad_mask"] & batch["available"]
    if int(mask.sum()) == 0:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise = torch.randn(velocity.shape, generator=gen).to(velocity.device)
    x0 = torch.where(mask, (velocity - mu) / sigma, torch.zeros_like(velocity))
    losses = []
    for t_val in timesteps:
        t = torch.full((velocity.size(0),), int(t_val), device=velocity.device, dtype=torch.long)
        x_t = model.schedule.q_sample(x0, t, noise)
        x_t = torch.where(mask, x_t, torch.zeros_like(x_t))
        eps = model(
            batch["pitch"], batch["timing"], batch["voice"], x_t, mask, t, batch["pad_mask"]
        )
        losses.append(F.mse_loss(eps[mask], noise[mask]).item())
    return float(sum(losses) / len(losses))
