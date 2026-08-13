"""Bundled data contracts for canonical SupplyChain-JEPA episodes."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def canonical_episode_schema() -> Traversable:
    """Return the packaged JSON Schema for raw canonical episodes."""

    return files(__package__).joinpath("canonical_episode.schema.json")


__all__ = ["canonical_episode_schema"]
