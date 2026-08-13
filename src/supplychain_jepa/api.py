"""Small, shape-explicit inference wrappers over the released neural core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .bundle import ModelBundle
from .model import SequenceEncoding


@dataclass(frozen=True)
class ActionComparison:
    """Latent futures for candidate actions, without pretending to be a policy.

    ``predicted_latents`` has shape ``[B,K,Q,D]``. ``pairwise_distances`` has
    shape ``[B,K,K,Q]`` and contains Euclidean distances between candidate
    latent futures. A downstream cost or value head is still required to rank
    candidates by operational preference.
    """

    predicted_latents: Tensor
    pairwise_distances: Tensor


def encode_state(
    bundle: ModelBundle,
    batch: Mapping[str, Any],
    *,
    prefix: str = "context",
) -> SequenceEncoding:
    """Encode token tensors into ``[B,L,D]`` tokens and two ``[B,D]`` states."""

    if prefix not in {"context", "target"}:
        raise ValueError("prefix must be 'context' or 'target'")
    with torch.no_grad():
        return bundle.model.online_encoder.encode_with_special_tokens(batch, prefix)


def predict_masked_state(
    bundle: ModelBundle,
    batch: Mapping[str, Any],
    *,
    role_ids: Tensor | None = None,
    horizons: Tensor | None = None,
    target_padding_mask: Tensor | None = None,
) -> Tensor:
    """Predict target token latents with output shape ``[B,Q,D]``.

    The batch must contain complete ``context_*`` tensors and target metadata
    tensors. Target values are not read when constructing prediction queries.
    """

    with torch.no_grad():
        return bundle.model.predict_state(
            batch,
            role_ids=role_ids,
            horizons=horizons,
            target_padding_mask=target_padding_mask,
        )


def predict_dynamics(
    bundle: ModelBundle,
    state_latents: Tensor,
    actions: Tensor,
    horizons: Tensor,
    *,
    state_padding_mask: Tensor | None = None,
    action_padding_mask: Tensor | None = None,
    exogenous: Tensor | None = None,
) -> Tensor:
    """Predict action-conditioned latent futures with shape ``[B,Q,D]``.

    Inputs must be token/state memory ``[B,S,D]``, actions ``[B,A,U]``, and
    horizons ``[B,Q]``. ``U`` must equal the checkpoint action dimension.
    """

    with torch.no_grad():
        return bundle.model.dynamics_predictor(
            state_latents,
            actions,
            horizons,
            state_padding_mask=state_padding_mask,
            action_padding_mask=action_padding_mask,
            exogenous=exogenous,
        )


def compare_actions(
    bundle: ModelBundle,
    state_latents: Tensor,
    candidate_actions: Tensor,
    horizons: Tensor,
    *,
    state_padding_mask: Tensor | None = None,
    action_padding_mask: Tensor | None = None,
    exogenous: Tensor | None = None,
) -> ActionComparison:
    """Evaluate ``K`` action sequences from each state in one batched call.

    ``state_latents`` is ``[B,S,D]`` and ``candidate_actions`` is ``[B,K,A,U]``.
    ``horizons`` may be shared across candidates as ``[B,Q]`` or supplied per
    candidate as ``[B,K,Q]``. Optional masks and exogenous inputs follow the
    same shared-or-per-candidate convention.
    """

    if state_latents.ndim != 3:
        raise ValueError("state_latents must be [B,S,D]")
    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must be [B,K,A,U]")
    batch_size, candidates, action_count, action_dim = candidate_actions.shape
    if state_latents.shape[0] != batch_size:
        raise ValueError("state and candidate batch dimensions do not match")
    expected_action_dim = bundle.model.config.action_dim
    if action_dim != expected_action_dim:
        raise ValueError(f"Expected candidate action dimension {expected_action_dim}")

    repeated_state = state_latents[:, None].expand(-1, candidates, -1, -1).reshape(
        batch_size * candidates, state_latents.shape[1], state_latents.shape[2]
    )
    flat_actions = candidate_actions.reshape(
        batch_size * candidates, action_count, action_dim
    )

    def _expand(value: Tensor | None, shared_ndim: int, name: str) -> Tensor | None:
        if value is None:
            return None
        if value.ndim == shared_ndim:
            if value.shape[0] != batch_size:
                raise ValueError(f"{name} batch dimension does not match")
            value = value[:, None].expand(-1, candidates, *value.shape[1:])
        elif value.ndim != shared_ndim + 1 or value.shape[:2] != (
            batch_size,
            candidates,
        ):
            raise ValueError(f"{name} has an invalid shared/candidate shape")
        return value.reshape(batch_size * candidates, *value.shape[2:])

    flat_horizons = _expand(horizons, 2, "horizons")
    assert flat_horizons is not None
    flat_state_mask = _expand(state_padding_mask, 2, "state_padding_mask")
    flat_action_mask = _expand(action_padding_mask, 2, "action_padding_mask")
    flat_exogenous = _expand(exogenous, 3, "exogenous")

    predicted = predict_dynamics(
        bundle,
        repeated_state,
        flat_actions,
        flat_horizons,
        state_padding_mask=flat_state_mask,
        action_padding_mask=flat_action_mask,
        exogenous=flat_exogenous,
    ).reshape(batch_size, candidates, flat_horizons.shape[1], -1)
    pairwise = torch.linalg.vector_norm(
        predicted[:, :, None] - predicted[:, None, :], dim=-1
    ).permute(0, 1, 2, 3)
    return ActionComparison(predicted_latents=predicted, pairwise_distances=pairwise)
