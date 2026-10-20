"""Training support search and explicitly held-out validation selection."""

import jax
import jax.numpy as jnp
import numpy as np
from .fastol import FastOL, TrainingNormalizer


@jax.jit
def _support_count(distances, epsilon):
    return 1 - jnp.count_nonzero(epsilon * distances < 1) / distances.size


def epsilon_for_sparsity(
    normalized_training,
    target_sparsity,
    *,
    tolerance=1e-3,
    max_iterations=60,
    initial_upper=1.0,
):
    """Ram's train-matrix support-count bisection, not validation/test selection."""
    x = jnp.asarray(normalized_training)
    if x.ndim != 2 or not len(x) or not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError("training inputs must be finite and nonempty")
    if not np.isfinite(target_sparsity) or not 0 <= target_sparsity <= 1 - 1 / len(x):
        raise ValueError("target sparsity exceeds the diagonal-only limit")
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or int(max_iterations) != max_iterations
        or max_iterations < 1
        or not np.isfinite(initial_upper)
        or initial_upper <= 0
    ):
        raise ValueError("invalid sparsity search settings")
    d = jnp.sqrt(
        jnp.maximum(
            jnp.sum(x * x, axis=1)[:, None]
            + jnp.sum(x * x, axis=1)[None, :]
            - 2 * jnp.matmul(x, x.T, precision=jax.lax.Precision.HIGHEST),
            0,
        )
    )

    def sparsity(ep):
        return float(_support_count(d, ep))

    if target_sparsity == 0:
        maxd = float(jnp.max(d))
        ep = 0.5 / maxd if maxd > 0 else 1.0
        return ep, sparsity(ep)
    maximum = 1 - float(jnp.count_nonzero(d == 0)) / d.size
    if target_sparsity > maximum + tolerance:
        raise ValueError(
            "duplicate/zero-distance inputs make target sparsity unattainable"
        )
    low = 0.0
    high = float(initial_upper)
    high_sparsity = sparsity(high)
    while high_sparsity < target_sparsity:
        high *= 2
        if not np.isfinite(high):
            raise ValueError("could not bracket target sparsity")
        high_sparsity = sparsity(high)
    best, best_sparsity, gap = high, high_sparsity, abs(high_sparsity - target_sparsity)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        s = sparsity(mid)
        delta = abs(s - target_sparsity)
        if delta < gap:
            best, best_sparsity, gap = mid, s, delta
        if delta <= tolerance:
            break
        if s < target_sparsity:
            low = mid
        else:
            high = mid
    return best, best_sparsity


def select_fastol_support(
    train_inputs,
    points,
    train_velocity,
    validation_inputs,
    validation_velocity,
    target_sparsities,
    *,
    train_ids=None,
    validation_ids=None,
    **model_options,
):
    """Select support using validation predictions; return epsilon and records.

    Test inputs/targets are deliberately absent from this API. Splits remain
    caller-owned; supply example IDs to check train/validation disjointness.
    Relative vector L2 uses all provided spatial/time/component entries.
    The final model is fit separately using the returned training-derived epsilon.
    """
    if "epsilon" in model_options:
        raise ValueError("epsilon is selected here, not supplied")
    if (train_ids is None) != (validation_ids is None):
        raise ValueError("supply both train_ids and validation_ids")
    if train_ids is not None:
        if len(train_ids) != len(train_inputs) or len(validation_ids) != len(
            validation_inputs
        ):
            raise ValueError("split IDs must match example counts")
        if np.intersect1d(train_ids, validation_ids).size:
            raise ValueError("training and validation IDs overlap")
    truth = jnp.asarray(validation_velocity)
    if truth.shape[0] != len(validation_inputs) or not bool(
        jnp.all(jnp.isfinite(truth))
    ):
        raise ValueError("invalid validation targets")
    denom = float(jnp.linalg.norm(truth))
    if denom == 0:
        raise ValueError("relative validation error requires nonzero targets")
    normalized = TrainingNormalizer.fit(train_inputs).transform(train_inputs)
    records = []
    for target in target_sparsities:
        ep, achieved = epsilon_for_sparsity(normalized, float(target))
        with FastOL(
            train_inputs, points, train_velocity, epsilon=ep, **model_options
        ) as model:
            prediction = model.predict(validation_inputs)
            if prediction.shape != truth.shape:
                raise ValueError("validation target shape does not match predictions")
            error = float(jnp.linalg.norm(prediction - truth)) / denom
        if not np.isfinite(error):
            raise RuntimeError("nonfinite validation score")
        records.append(
            {
                "target_sparsity": float(target),
                "achieved_sparsity": achieved,
                "epsilon": ep,
                "validation_relative_l2": error,
            }
        )
    if not records:
        raise ValueError("at least one support candidate is required")
    best = min(records, key=lambda r: r["validation_relative_l2"])
    return best.copy(), records
