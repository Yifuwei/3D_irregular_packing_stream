"""CA defaults selected from the September 2026 cross-dataset experiment.

Use the user's selected combinations, which maximize mean U* by bin shape.
These are empirical defaults across 15 datasets, not global optima.
Only PES and selection range were compared; other settings stay unchanged.
"""


def best_ca_parameters(container_shape):
    """Return (SCH_nesting_strategy, selection_range) for a bin shape."""
    if container_shape == "cube":
        return "maximal_residual_box", "bottom"
    if container_shape == "cylinder":
        return "minimum_aabb_edges_len", "bottom_top"
    raise ValueError(f"Unsupported container shape: {container_shape}")
