"""DDPM noise schedule and single-step reverse updates (epsilon parameterization)."""

from __future__ import annotations

import torch
from torch import nn


class DDPMSchedule(nn.Module):
    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bar[:-1]])

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ab)*x0 + sqrt(1-ab)*noise."""
        sa = self.sqrt_alpha_bar[t].view(-1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        return sa * x0 + so * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sa = self.sqrt_alpha_bar[t].view(-1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        return (x_t - so * eps) / sa.clamp_min(1e-8)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        eps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """One reverse DDPM step from x_t to x_{t-1}."""
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        mean = self.sqrt_recip_alphas[t] * (
            x_t - beta / self.sqrt_one_minus_alpha_bar[t] * eps
        )
        if t == 0:
            return mean
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return mean + self.posterior_variance[t].sqrt() * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eps: torch.Tensor,
        *,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDIM update from index t to t_prev (both in [0, timesteps-1]); eta=0 is deterministic."""
        ab = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t_prev] if t_prev >= 0 else x_t.new_tensor(1.0)
        pred_x0 = (x_t - self.sqrt_one_minus_alpha_bar[t] * eps) / self.sqrt_alpha_bar[t].clamp_min(
            1e-8
        )
        if t_prev < 0:
            return pred_x0
        sigma = (
            eta
            * (
                (1 - ab_prev)
                / (1 - ab).clamp_min(1e-8)
                * (1 - ab / ab_prev.clamp_min(1e-8))
            ).clamp_min(0.0).sqrt()
        )
        dir_eps = (1 - ab_prev - sigma**2).clamp_min(0.0).sqrt() * eps
        x_prev = ab_prev.sqrt() * pred_x0 + dir_eps
        if eta > 0 and sigma > 0:
            noise = torch.randn(
                x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
            )
            x_prev = x_prev + sigma * noise
        return x_prev
