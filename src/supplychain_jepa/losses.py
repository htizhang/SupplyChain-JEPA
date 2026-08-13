"""Separately callable latent objectives and collapse diagnostics."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def _expanded_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.shape == reference.shape:
        return mask.bool()
    if mask.shape == reference.shape[:-1]:
        return mask.bool().unsqueeze(-1).expand_as(reference)
    raise ValueError("mask must match latent shape or latent shape without D")


def masked_latent_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    kind: Literal["smooth_l1", "l2"] = "smooth_l1",
    beta: float = 1.0,
) -> Tensor:
    """Masked Smooth-L1 or elementwise L2 latent prediction loss."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    selected = _expanded_mask(mask, prediction)
    if not selected.any():
        raise ValueError("loss mask selects no elements")
    if kind == "smooth_l1":
        values = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    elif kind == "l2":
        values = (prediction - target).square()
    else:
        raise ValueError(f"Unsupported latent loss {kind!r}")
    return values.masked_select(selected).mean()


def _flatten_valid(latents: Tensor, mask: Tensor | None) -> Tensor:
    if latents.ndim < 2:
        raise ValueError("latents must have a final embedding dimension")
    rows = latents.reshape(-1, latents.shape[-1])
    if mask is None:
        return rows
    expected = latents.shape[:-1]
    if mask.shape != expected:
        raise ValueError(f"mask must have shape {expected}")
    return rows[mask.reshape(-1).bool()]


def vicreg_variance_loss(
    latents: Tensor,
    *,
    mask: Tensor | None = None,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> Tensor:
    """VICReg hinge penalty for dimensions whose batch standard deviation is low."""

    rows = _flatten_valid(latents, mask)
    if rows.shape[0] < 2:
        raise ValueError("variance loss requires at least two valid latent rows")
    std = torch.sqrt(rows.var(dim=0, unbiased=False) + eps)
    return F.relu(float(target_std) - std).mean()


def vicreg_covariance_loss(
    latents: Tensor,
    *,
    mask: Tensor | None = None,
    normalization: Literal["legacy_pair_mean", "standard_width"] = "legacy_pair_mean",
) -> Tensor:
    """Squared off-diagonal covariance with an explicit normalization contract."""

    rows = _flatten_valid(latents, mask)
    if rows.shape[0] < 2:
        raise ValueError("covariance loss requires at least two valid latent rows")
    centered = rows - rows.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (rows.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal_square = covariance.square().sum() - diagonal.square().sum()
    width = covariance.shape[0]
    if width < 2:
        return off_diagonal_square
    if normalization == "legacy_pair_mean":
        return off_diagonal_square / (width * (width - 1))
    if normalization == "standard_width":
        return off_diagonal_square / width
    raise ValueError(f"Unsupported covariance normalization {normalization!r}")


def sigreg_loss(
    latents: Tensor,
    *,
    num_projections: int = 256,
    knots: int = 17,
    max_frequency: float = 3.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sketch Isotropic Gaussian Regularization over independent latent rows.

    This is the Epps--Pulley random-projection objective used by LeJEPA/LeWM,
    adapted to a simple ``[N, D]`` contract.  Computation is promoted to
    float32 so AMP does not destabilize the characteristic-function estimate.
    The random directions are always allocated on the latent tensor's device.
    """

    rows = _flatten_valid(latents, None)
    if rows.shape[0] < 2:
        raise ValueError("SIGReg requires at least two independent latent rows")
    if num_projections < 1:
        raise ValueError("num_projections must be positive")
    if knots < 2:
        raise ValueError("knots must be at least two")
    if max_frequency <= 0:
        raise ValueError("max_frequency must be positive")

    rows = rows.float()
    directions = torch.randn(
        rows.shape[-1],
        int(num_projections),
        device=rows.device,
        dtype=rows.dtype,
        generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    frequencies = torch.linspace(
        0.0,
        float(max_frequency),
        int(knots),
        device=rows.device,
        dtype=rows.dtype,
    )
    delta = float(max_frequency) / (int(knots) - 1)
    trapezoid = torch.full_like(frequencies, 2.0 * delta)
    trapezoid[[0, -1]] = delta
    gaussian_characteristic = torch.exp(-frequencies.square() / 2.0)
    weights = trapezoid * gaussian_characteristic

    projected = rows @ directions
    phase = projected.unsqueeze(-1) * frequencies
    error = (
        (phase.cos().mean(dim=0) - gaussian_characteristic).square()
        + phase.sin().mean(dim=0).square()
    )
    statistic = (error @ weights) * rows.shape[0]
    return statistic.mean()


def paired_difference_loss(
    prediction_a: Tensor,
    prediction_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    *,
    mask: Tensor | None = None,
    kind: Literal["smooth_l1", "l2"] = "smooth_l1",
    beta: float = 1.0,
) -> Tensor:
    """Compare predicted and true intervention-induced latent differences."""

    if not (
        prediction_a.shape
        == prediction_b.shape
        == target_a.shape
        == target_b.shape
    ):
        raise ValueError("All paired tensors must have identical shapes")
    return masked_latent_loss(
        prediction_b - prediction_a,
        target_b - target_a,
        mask=mask,
        kind=kind,
        beta=beta,
    )


def inverse_action_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    kind: Literal["classification", "regression"] = "classification",
    beta: float = 1.0,
) -> Tensor:
    """Optional inverse-dynamics component for discrete or continuous actions."""

    if kind == "classification":
        if prediction.ndim != target.ndim + 1:
            raise ValueError("classification logits require one final class dimension")
        logits = prediction.reshape(-1, prediction.shape[-1])
        labels = target.reshape(-1).long()
        if mask is not None:
            if mask.shape != target.shape:
                raise ValueError("classification mask must match target")
            selected = mask.reshape(-1).bool()
            if not selected.any():
                raise ValueError("loss mask selects no actions")
            logits, labels = logits[selected], labels[selected]
        return F.cross_entropy(logits, labels)
    if kind == "regression":
        return masked_latent_loss(
            prediction, target, mask=mask, kind="smooth_l1", beta=beta
        )
    raise ValueError(f"Unsupported inverse-action loss {kind!r}")


@torch.no_grad()
def collapse_diagnostics(
    latents: Tensor,
    *,
    mask: Tensor | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return mean std, effective rank, and leading-PC variance fraction."""

    rows = _flatten_valid(latents.detach().float(), mask)
    if rows.shape[0] < 2:
        return {
            "sample_count": float(rows.shape[0]),
            "mean_std": 0.0,
            "effective_rank": 0.0,
            "top_pc_ratio": 1.0,
        }
    centered = rows - rows.mean(dim=0, keepdim=True)
    std = rows.std(dim=0, unbiased=False).mean()
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    total = variance.sum()
    if float(total) <= eps:
        effective_rank = 0.0
        top_pc_ratio = 1.0
    else:
        probabilities = variance / total
        positive = probabilities > eps
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(torch.exp(entropy))
        top_pc_ratio = float(probabilities.max())
    return {
        "sample_count": float(rows.shape[0]),
        "mean_std": float(std),
        "effective_rank": effective_rank,
        "top_pc_ratio": top_pc_ratio,
    }
