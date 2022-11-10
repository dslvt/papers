import math

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from PIL import Image


def get_random_string(alphabet: np.ndarray, ln: int, rnd: jax.random.PRNGKey) -> str:
    s = "".join(
        alphabet[
            jax.random.randint(
                rnd,
                shape=(1, ln),
                minval=0,
                maxval=alphabet.shape[0],
            )
        ][0]
    )
    return s


def save_image(
    ndarray: jnp.ndarray,
    fp: str,
    nrow: int = 8,
    padding: int = 2,
    pad_value: float = 0.0,
    format: str = None,
) -> None:
    if not (
        isinstance(ndarray, jnp.ndarray)
        or (
            isinstance(ndarray, list)
            and all(isinstance(t, jnp.ndarray) for t in ndarray)
        )
    ):
        raise TypeError(f"array_like of tensors expected, got {type(ndarray)}")

    ndarray = jnp.array(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels), pad_value
    ).astype(jnp.float32)
    k = 0

    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break

            grid = grid.at[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ].set(ndarray[k])
            k += 1

    ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)


def str_tree(d, depth):
    s = ""
    for k in d.keys():
        if isinstance(d[k], FrozenDict):
            s += f'{"  " * depth} {k}\n'
            s += str_tree(d[k], depth + 1)
        else:
            s += f'{"  " * depth} {k} {d[k].shape}\n'
    return s
