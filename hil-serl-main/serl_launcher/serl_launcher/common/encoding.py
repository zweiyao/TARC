from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        temporal_key: Observation key holding a (..., T, C) time series. None
            disables the branch entirely.
        temporal_encoder: Module encoding that series into a flat vector.
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    temporal_key: Optional[str] = None
    temporal_encoder: Optional[nn.Module] = None
    temporal_stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)

        if self.temporal_key is not None:
            assert (
                self.temporal_encoder is not None
            ), "temporal_key is set but temporal_encoder is None"
            series = observations[self.temporal_key]
            if self.enable_stacking:
                # Fold the obs-stacking axis into the time axis
                if len(series.shape) == 3:
                    series = rearrange(series, "S T C -> (S T) C")
                    encoded = encoded.reshape(-1)
                if len(series.shape) == 4:
                    series = rearrange(series, "B S T C -> B (S T) C")

            series = self.temporal_encoder(series, train=train)

            if stop_gradient and self.temporal_stop_gradient:
                series = jax.lax.stop_gradient(series)

            encoded = jnp.concatenate([encoded, series], axis=-1)

        return encoded
