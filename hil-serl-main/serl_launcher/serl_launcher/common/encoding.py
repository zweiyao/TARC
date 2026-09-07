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
        tactile_keys: Keys encoded by the same shared ResNet trunk as the
            cameras, but kept out of image_keys so they skip the crop
            augmentation and the replay buffer's frame reuse. Their encoders
            carry their own normalize/do_resize settings.
        action_chunk_key: Observation key holding a (..., T, C) time series —
            currently the pi0.5 action chunk. None disables the branch.
            Named for its caller, not its shape: the encoder underneath is a
            generic series encoder and would take any (..., T, C).
        action_chunk_encoder: Module encoding that series into a flat vector.
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    tactile_keys: Iterable[str] = ()
    action_chunk_key: Optional[str] = None
    action_chunk_encoder: Optional[nn.Module] = None
    action_chunk_stop_gradient: bool = False

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

        # Same encoder dict, so the frozen trunk stays shared; what differs is
        # the per-head normalize/do_resize set in sac.py. Placed before the
        # concat so the reshape(-1) in the branches below flattens it too on the
        # no-batch inference path.
        for tactile_key in self.tactile_keys:
            tactile = observations[tactile_key]
            if not is_encoded and self.enable_stacking:
                if len(tactile.shape) == 4:
                    tactile = rearrange(tactile, "T H W C -> H W (T C)")
                if len(tactile.shape) == 5:
                    tactile = rearrange(tactile, "B T H W C -> B H W (T C)")

            tactile = self.encoder[tactile_key](
                tactile, train=train, encode=not is_encoded
            )

            if stop_gradient:
                tactile = jax.lax.stop_gradient(tactile)

            encoded.append(tactile)

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

        if self.action_chunk_key is not None:
            assert (
                self.action_chunk_encoder is not None
            ), "action_chunk_key is set but action_chunk_encoder is None"
            series = observations[self.action_chunk_key]
            if self.enable_stacking:
                # Fold the obs-stacking axis into the time axis
                if len(series.shape) == 3:
                    series = rearrange(series, "S T C -> (S T) C")
                    encoded = encoded.reshape(-1)
                if len(series.shape) == 4:
                    series = rearrange(series, "B S T C -> B (S T) C")

            series = self.action_chunk_encoder(series, train=train)

            if stop_gradient and self.action_chunk_stop_gradient:
                series = jax.lax.stop_gradient(series)

            encoded = jnp.concatenate([encoded, series], axis=-1)

        return encoded
