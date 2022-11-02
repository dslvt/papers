import jax
import numpy as np


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
