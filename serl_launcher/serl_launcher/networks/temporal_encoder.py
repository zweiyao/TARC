from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class TemporalConvEncoder(nn.Module):
    """Encodes a fixed-length time series (..., T, C) into a flat (..., latent_dim).

    A small 1D CNN runs along the time axis, so the first kernel spans several
    consecutive steps instead of flattening time into the feature dimension.
    Global pooling makes the output independent of T.

    The output head (Dense -> LayerNorm -> tanh) mirrors the proprio branch of
    EncodingWrapper and the ResNet bottleneck, so all branches are on the same
    scale before they are concatenated.
    """

    latent_dim: int = 64
    hidden_dims: Sequence[int] = (64, 64)
    kernel_size: int = 5
    strides: int = 1
    pooling: str = "mean_max"  # "mean", "max", "mean_max" or "last"
    use_layer_norm: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] | str = nn.tanh
    # Critic.__call__ does not forward `train`, so dropout would make the actor
    # and critic passes disagree. Left off by default.
    dropout_rate: Optional[float] = None
    # Whatever series is fed in may be orders of magnitude above 1 and would
    # saturate the tanh. pi0.5 action chunks are already roughly in [-1, 1], so
    # 1.0 is right for the current caller; raw sensor readings would not be.
    input_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        assert x.ndim >= 2, f"expected (..., T, C), got shape {x.shape}"
        lead_shape = x.shape[:-2]
        time_steps, channels = x.shape[-2:]

        # Collapse whatever leading dims we got (none, batch, or batch+stack)
        # so one code path serves both inference (1, T, C) and training
        # (B, 1, T, C).
        h = x.astype(jnp.float32).reshape((-1, time_steps, channels))
        h = h / self.input_scale

        for i, features in enumerate(self.hidden_dims):
            h = nn.Conv(
                features=features,
                kernel_size=(self.kernel_size,),
                strides=(self.strides,),
                padding="SAME",
                kernel_init=default_init(),
                name=f"conv_{i}",
            )(h)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"ln_{i}")(h)
            h = activations(h)

        if self.pooling == "mean":
            pooled = h.mean(axis=-2)
        elif self.pooling == "max":
            pooled = h.max(axis=-2)
        elif self.pooling == "mean_max":
            # mean alone drops contact spikes, max alone drops the overall level
            pooled = jnp.concatenate([h.mean(axis=-2), h.max(axis=-2)], axis=-1)
        elif self.pooling == "last":
            pooled = h[..., -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        z = nn.Dense(self.latent_dim, kernel_init=default_init(), name="proj")(pooled)
        z = nn.LayerNorm(name="ln_out")(z)
        z = nn.tanh(z)

        return z.reshape(lead_shape + (self.latent_dim,))
