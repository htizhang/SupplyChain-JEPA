"""Deterministic structured objective masks for State JEPA."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .data import EpisodeTensors

MASK_KINDS = (
    "contiguous_time",
    "entity_group",
    "network_neighborhood",
    "event_suffix",
    "outage",
    "future_horizon",
)


def _seed_from(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass
class MaskPlan:
    """One objective mask and the corresponding leakage-safe context view."""

    kind: str
    objective_target_token_mask: Tensor
    objective_target_feature_mask: Tensor
    context_token_keep_mask: Tensor
    context_feature_keep_mask: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredMaskSampler:
    """Sample deterministic supply-chain structured masks.

    The sampler never mutates the episode. Every plan carries both the target
    objective and a context keep-mask. Any legally observed feature aligned
    with a selected target is removed from context. Future-horizon plans also
    remove every future context token, including future actions.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        min_time_span: int = 1,
        max_time_span: int = 4,
        max_entity_group: int = 4,
        neighborhood_radius: int = 1,
        future_horizons: tuple[int, ...] = (1, 2, 4),
        kinds: tuple[str, ...] = MASK_KINDS,
    ):
        if min_time_span < 1 or max_time_span < min_time_span:
            raise ValueError("Invalid time-span bounds")
        if max_entity_group < 1:
            raise ValueError("max_entity_group must be positive")
        if neighborhood_radius < 0:
            raise ValueError("neighborhood_radius must be non-negative")
        if not future_horizons or any(horizon < 1 for horizon in future_horizons):
            raise ValueError("future_horizons must contain positive endpoint distances")
        unknown = set(kinds) - set(MASK_KINDS)
        if unknown:
            raise ValueError(f"Unknown mask kinds: {sorted(unknown)}")
        self.seed = int(seed)
        self.min_time_span = int(min_time_span)
        self.max_time_span = int(max_time_span)
        self.max_entity_group = int(max_entity_group)
        self.neighborhood_radius = int(neighborhood_radius)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.kinds = tuple(kinds)

    @staticmethod
    def _times(episode: EpisodeTensors) -> list[int]:
        return sorted(
            {
                int(value)
                for value in episode.target_time_index.tolist()
                if int(value) >= 0
            }
        )

    @staticmethod
    def _token_mask(episode: EpisodeTensors, indices: Iterable[int]) -> Tensor:
        mask = torch.zeros((episode.target_length,), dtype=torch.bool)
        selected = list(indices)
        if selected:
            mask[selected] = True
        return mask

    def _contiguous_time(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        times = self._times(episode)
        span = min(
            len(times),
            rng.randint(self.min_time_span, min(self.max_time_span, len(times))),
        )
        start_index = rng.randint(0, len(times) - span)
        selected_times = set(times[start_index : start_index + span])
        indices = [
            index
            for index, time_index in enumerate(episode.target_time_index.tolist())
            if int(time_index) in selected_times
        ]
        return self._token_mask(episode, indices), {
            "start": min(selected_times),
            "end": max(selected_times),
        }

    def _entity_group(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        entities = sorted(
            {
                int(entity)
                for kind, entity in zip(
                    episode.target_token_kinds,
                    episode.target_entity_index.tolist(), strict=False,
                )
                if kind == "entity" and int(entity) > 0
            }
        )
        if not entities:
            return self._contiguous_time(episode, rng)
        count = rng.randint(1, min(self.max_entity_group, len(entities)))
        selected = set(rng.sample(entities, count))
        indices = [
            index
            for index, entity in enumerate(episode.target_entity_index.tolist())
            if int(entity) in selected
        ]
        return self._token_mask(episode, indices), {"entities": sorted(selected)}

    def _neighborhood_nodes(
        self, episode: EpisodeTensors, root: int
    ) -> set[int]:
        selected = {root}
        frontier = {root}
        for _ in range(self.neighborhood_radius):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(episode.node_neighbors.get(node, ()))
            next_frontier -= selected
            selected.update(next_frontier)
            frontier = next_frontier
        return selected

    def _network_neighborhood(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        candidates = sorted(
                node
                for node in episode.node_neighbors
                if any(
                    node in (node_key, source_key, target_key)
                    for node_key, source_key, target_key in zip(
                    episode.target_node_index.tolist(),
                    episode.target_source_index.tolist(),
                    episode.target_target_index.tolist(), strict=False,
                )
            )
        )
        if not candidates:
            return self._entity_group(episode, rng)
        root = rng.choice(candidates)
        selected = self._neighborhood_nodes(episode, root)
        indices = [
            index
            for index, (node, source, target) in enumerate(
                zip(
                    episode.target_node_index.tolist(),
                    episode.target_source_index.tolist(),
                    episode.target_target_index.tolist(), strict=False,
                )
            )
            if node in selected or source in selected or target in selected
        ]
        if not indices:
            return self._entity_group(episode, rng)
        return self._token_mask(episode, indices), {
            "root": root,
            "radius": self.neighborhood_radius,
            "nodes": sorted(selected),
        }

    def _event_suffix(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        event_times = sorted(
            {
                int(time_index)
                for kind, time_index in zip(
                    episode.target_token_kinds, episode.target_time_index.tolist(), strict=False
                )
                if kind == "event" and int(time_index) >= 0
            }
        )
        if not event_times:
            return self._contiguous_time(episode, rng)
        cutoff_position = rng.randrange(len(event_times))
        cutoff = event_times[cutoff_position]
        available = len(event_times) - cutoff_position
        span = rng.randint(1, min(4, self.max_time_span, available))
        selected_times = set(
            event_times[cutoff_position : cutoff_position + span]
        )
        indices = [
            index
            for index, (kind, time_index) in enumerate(
                zip(episode.target_token_kinds, episode.target_time_index.tolist(), strict=False)
            )
            if kind == "event" and int(time_index) in selected_times
        ]
        return self._token_mask(episode, indices), {
            "cutoff": cutoff,
            "end": max(selected_times),
            "window_steps": span,
            "window_times": sorted(selected_times),
        }

    def _outage(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        candidates = sorted(
            {
                int(node)
                for node in episode.target_node_index.tolist()
                if int(node) > 0
            }
        )
        times = self._times(episode)
        if not candidates or not times:
            return self._contiguous_time(episode, rng)
        node = rng.choice(candidates)
        span = min(len(times), rng.randint(1, min(self.max_time_span, len(times))))
        start_index = rng.randint(0, len(times) - span)
        selected_times = set(times[start_index : start_index + span])
        indices = [
            index
            for index, (token_node, time_index) in enumerate(
                zip(
                    episode.target_node_index.tolist(),
                    episode.target_time_index.tolist(), strict=False,
                )
            )
            if int(token_node) == node and int(time_index) in selected_times
        ]
        if not indices:
            return self._entity_group(episode, rng)
        return self._token_mask(episode, indices), {
            "node": node,
            "start": min(selected_times),
            "end": max(selected_times),
        }

    def _future_horizon(
        self, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        times = self._times(episode)
        if len(times) < 2:
            return self._contiguous_time(episode, rng)
        cutoff_position = rng.randrange(len(times) - 1)
        cutoff = times[cutoff_position]
        available = len(times) - cutoff_position - 1
        possible = [value for value in self.future_horizons if value <= available]
        horizon = rng.choice(possible or (available,))
        selected_times = set(
            times[cutoff_position + 1 : cutoff_position + 1 + horizon]
        )
        end = max(selected_times)
        indices = [
            index
            for index, time_index in enumerate(episode.target_time_index.tolist())
            if int(time_index) in selected_times
        ]
        return self._token_mask(episode, indices), {
            "cutoff": cutoff,
            "horizon": horizon,
            "end": end,
            "window_times": sorted(selected_times),
            "action_omitted": True,
            "interpretation": "policy_marginal",
        }

    def _select(
        self, kind: str, episode: EpisodeTensors, rng: random.Random
    ) -> tuple[Tensor, dict[str, Any]]:
        return {
            "contiguous_time": self._contiguous_time,
            "entity_group": self._entity_group,
            "network_neighborhood": self._network_neighborhood,
            "event_suffix": self._event_suffix,
            "outage": self._outage,
            "future_horizon": self._future_horizon,
        }[kind](episode, rng)

    @staticmethod
    def _redact_aligned_context(
        episode: EpisodeTensors,
        objective_feature_mask: Tensor,
        context_feature_keep: Tensor,
    ) -> None:
        selected_by_alignment: dict[int, Tensor] = {}
        for index, alignment in enumerate(episode.target_alignment_index.tolist()):
            selected = objective_feature_mask[index]
            if selected.any():
                alignment = int(alignment)
                if alignment in selected_by_alignment:
                    selected_by_alignment[alignment] |= selected
                else:
                    selected_by_alignment[alignment] = selected.clone()
        for index, alignment in enumerate(episode.context_alignment_index.tolist()):
            selected = selected_by_alignment.get(int(alignment))
            if selected is not None:
                context_feature_keep[index] &= ~selected

    def sample(
        self,
        episode: EpisodeTensors,
        *,
        epoch: int = 0,
        kind: str | None = None,
    ) -> MaskPlan:
        if episode.target_length == 0:
            raise ValueError("Cannot mask an episode with no targets")
        selected_kind = kind
        if selected_kind is None:
            selector = random.Random(
                _seed_from(self.seed, episode.record_id, epoch, "kind")
            )
            selected_kind = selector.choice(self.kinds)
        if selected_kind not in self.kinds:
            raise ValueError(f"Mask kind {selected_kind!r} is not enabled")
        rng = random.Random(
            _seed_from(self.seed, episode.record_id, epoch, selected_kind)
        )
        token_mask, metadata = self._select(selected_kind, episode, rng)
        objective_feature_mask = (
            token_mask.unsqueeze(1) & episode.target_feature_present
        )
        if not objective_feature_mask.any():
            # Every canonical latent token has at least one feature, but retain a
            # deterministic defensive fallback for future schema versions.
            token_mask, fallback = self._contiguous_time(episode, rng)
            objective_feature_mask = (
                token_mask.unsqueeze(1) & episode.target_feature_present
            )
            metadata = {"fallback_from": selected_kind, **fallback}

        if (
            selected_kind in {"future_horizon", "event_suffix"}
            and "cutoff" not in metadata
        ):
            raise ValueError(
                f"{selected_kind} mask is unavailable for this episode"
            )

        context_token_keep = torch.ones(
            (episode.context_length,), dtype=torch.bool
        )
        context_feature_keep = episode.context_feature_present.clone()
        self._redact_aligned_context(
            episode, objective_feature_mask, context_feature_keep
        )

        if selected_kind == "future_horizon":
            cutoff = int(metadata["cutoff"])
            future_context = episode.context_time_index > cutoff
            context_token_keep[future_context] = False
            context_feature_keep[future_context] = False
        elif selected_kind == "event_suffix":
            cutoff = int(metadata["cutoff"])
            suffix_window_events = torch.tensor(
                [
                    token_kind == "event"
                    and int(time_index) >= cutoff
                    for token_kind, time_index in zip(
                        episode.context_token_kinds,
                        episode.context_time_index.tolist(), strict=False,
                    )
                ],
                dtype=torch.bool,
            )
            context_token_keep[suffix_window_events] = False
            context_feature_keep[suffix_window_events] = False

        context_token_keep &= context_feature_keep.any(dim=1)
        return MaskPlan(
            kind=selected_kind,
            objective_target_token_mask=token_mask,
            objective_target_feature_mask=objective_feature_mask,
            context_token_keep_mask=context_token_keep,
            context_feature_keep_mask=context_feature_keep,
            metadata=metadata,
        )
