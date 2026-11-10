"""Shared CA defaults selected by mean N-U* across 15 datasets and seeds 3, 7, 13.

Both bin shapes use minimum AABB edge length and the bottom_top range.
These are empirical defaults across 15 datasets, not global optima.
Only PES and selection range were compared; other settings stay unchanged.
"""


def best_ca_parameters(container_shape):
    """Return (SCH_nesting_strategy, selection_range) for a bin shape."""
    if container_shape == "cube":
        return "minimum_aabb_edges_len", "bottom_top"
    if container_shape == "cylinder":
        return "minimum_aabb_edges_len", "bottom_top"
    raise ValueError(f"Unsupported container shape: {container_shape}")
