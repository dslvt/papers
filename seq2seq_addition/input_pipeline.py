from typing import Any, Dict, Generator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import string

Array = Any


class CharacterTable:
    def __init__(self, chars: str, max_len_query_digit: int = 3) -> None:
        self._chars = sorted(set(chars))
        self._char_indices = {ch: idx + 2 for idx, ch in enumerate(self._chars)}
        self._indices_char = {idx + 2: ch for idx, ch in enumerate(self._chars)}

        self._indices_char[self.pad_id] = "_"
        self._max_len_query_digit = max_len_query_digit

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self._chars) + 2

    @property
    def max_input_len(self) -> int:
        return self._max_len_query_digit * 2 + 2

    @property
    def max_output_len(self) -> int:
        return self._max_len_query_digit + 3

    @property
    def encoder_input_shape(self) -> Tuple[int, int, int]:
        return (1, self.max_input_len, self.vocab_size)

    @property
    def decoder_input_shape(self) -> Tuple[int, int, int]:
        return (1, self.max_output_len, self.vocab_size)

    def encode(self, inputs: str) -> np.ndarray:
        return np.array([self._char_indices[ch] for ch in inputs] + [self.eos_id])

    def decode(self, inputs: Array) -> str:
        chars = []
        for el in inputs.tolist():
            if el == self.eos_id:
                break
            chars.append(self._indices_char[el])
        chars = np.array(chars)
        return "".join(chars)

    def one_hot(self, tokens: np.ndarray) -> np.ndarray:
        vecs = np.zeros((tokens.size, self.vocab_size), dtype=np.float32)
        vecs[np.arange(tokens.size), tokens] = 1
        return vecs

    def encode_onehot(
        self, batch_inputs: Array, max_len: Optional[int] = None
    ) -> np.ndarray:
        if max_len is None:
            max_len = self.max_input_len

        def encode_str(s):
            tokens = self.encode(s)
            unpadded_len = len(tokens)
            if unpadded_len > max_len:
                raise ValueError(
                    f"Sequence is too long ({unpadded_len} > {max_len}): '{s}'"
                )
            tokens = np.pad(tokens, [(0, max_len - len(tokens))], mode="constant")
            return self.one_hot(tokens)

        return np.array([encode_str(s) for s in batch_inputs])

    def decode_onehot(self, batch_inputs: Array) -> np.ndarray:
        decode_inputs = lambda inputs: self.decode(
            inputs.argmax(axis=-1)
        )  # TODO: Why we using argmax?

        return np.array(list(map(decode_inputs, batch_inputs)))

    def generate_examples(
        self, num_examples: int, rnd_key: jax.random.PRNGKey
    ) -> Generator[Tuple[str, str], None, None]:
        for _ in range(num_examples):
            max_digit = pow(10, self._max_len_query_digit) - 1
            rnd_key, rnd_1 = jax.random.split(rnd_key)
            rnd_key, rnd_2 = jax.random.split(rnd_key)
            key = tuple(
                sorted(
                    (
                        jax.random.randint(rnd_1, shape=(1,), minval=0, maxval=99)[0],
                        jax.random.randint(
                            rnd_2, shape=(1,), minval=0, maxval=max_digit
                        )[0],
                    )
                )
            )
            inputs = f"{key[0]}+{key[1]}"
            outputs = f"={key[0]+key[1]}"
            yield (inputs, outputs)

    def get_batch(
        self, batch_size: int, rnd_key: jax.random.PRNGKey
    ) -> Dict[str, np.ndarray]:
        inputs, outputs = zip(*self.generate_examples(batch_size, rnd_key))

        return {
            "query": self.encode_onehot(inputs, self.max_input_len),
            "answer": self.encode_onehot(outputs, self.max_output_len),
        }


def mask_sequences(sequence_batch: Array, lengths: Array) -> Array:
    return sequence_batch * (
        lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1])[np.newaxis]
    )


def get_sequence_lengths(sequence_batch: Array, eos_id: int) -> Array:
    # sequence_batch.shape = (batch_size, seq_length, vocab_size)
    eos_row = sequence_batch[:, :, eos_id]
    eos_idx = jnp.argmax(eos_row, axis=-1)
    return jnp.where(
        eos_row[jnp.arange(eos_row.shape[0]), eos_idx],
        eos_idx + 1,
        sequence_batch.shape[1],
    )


class AdditionTaskCT(CharacterTable):
    def __init__(self, max_len_query_digit: int = 3) -> None:
        chars = "0123456789+= "
        super().__init__(chars, max_len_query_digit)

    @property
    def max_input_len(self) -> int:
        # The input has the form "digit1+digit2<eos>", so the max input length is
        # the length of two digits plus two tokens for "+" and the EOS token.
        return self._max_len_query_digit * 2 + 2

    @property
    def max_output_len(self) -> int:
        # The output has the form "=digit<eos>". If `digit` is the result of adding
        # two digits of max length x, then max length of `digit` is x+1.
        # Additionally, we require two more tokens for "=" and "<eos>".
        return self._max_len_query_digit + 3

    def generate_examples(
        self, num_examples: int, rnd_key: jax.random.PRNGKey
    ) -> Generator[Tuple[str, str], None, None]:
        for _ in range(num_examples):
            max_digit = pow(10, self._max_len_query_digit) - 1
            rnd_key, rnd_1 = jax.random.split(rnd_key)
            rnd_key, rnd_2 = jax.random.split(rnd_key)
            key = tuple(
                sorted(
                    (
                        jax.random.randint(rnd_1, shape=(1,), minval=0, maxval=99)[0],
                        jax.random.randint(
                            rnd_2, shape=(1,), minval=0, maxval=max_digit
                        )[0],
                    )
                )
            )
            inputs = f"{key[0]}+{key[1]}"
            outputs = f"={key[0]+key[1]}"
            yield (inputs, outputs)


class WordAdditionTaskCT(CharacterTable):
    def __init__(self, max_len_query_digit: int = 3) -> None:
        self.alphabet = np.array([ch for ch in string.ascii_lowercase])
        chars = f"{string.ascii_lowercase}+= "
        super().__init__(chars, max_len_query_digit)

    @property
    def max_input_len(self) -> int:
        # The input has the form "word1+word2<eos>", so the max input length is
        # the length of two digits plus two tokens for "+" and the EOS token.
        return self._max_len_query_digit * 2 + 2

    @property
    def max_output_len(self) -> int:
        # The output has the form "=digit<eos>". If `digit` is the result of adding
        # two digits of max length x, then max length of `digit` is x+1.
        # Additionally, we require two more tokens for "=" and "<eos>".
        return self._max_len_query_digit * 2 + 2

    def get_random_string(self, rnd):
        s = "".join(
            self.alphabet[
                jax.random.randint(
                    rnd,
                    shape=(1, self._max_len_query_digit),
                    minval=0,
                    maxval=self.alphabet.shape[0],
                )
            ][0]
        )
        return s

    def generate_examples(
        self, num_examples: int, rnd_key: jax.random.PRNGKey
    ) -> Generator[Tuple[str, str], None, None]:
        for _ in range(num_examples):
            rnd_key, rnd_1 = jax.random.split(rnd_key)
            rnd_key, rnd_2 = jax.random.split(rnd_key)
            key = (self.get_random_string(rnd_1), self.get_random_string(rnd_2))
            inputs = f"{key[0]}+{key[1]}"
            outputs = f"={key[0]+key[1]}"
            yield (inputs, outputs)


class WordDiffAdditionTaskCT(CharacterTable):
    def __init__(self, max_len_query_digit: int = 3) -> None:
        self.alphabet = np.array([ch for ch in string.ascii_lowercase])
        chars = f"{string.ascii_lowercase}+= "
        super().__init__(chars, max_len_query_digit)

    @property
    def max_input_len(self) -> int:
        # The input has the form "word1+word2<eos>", so the max input length is
        # the length of word plus one token for the EOS token.
        return self._max_len_query_digit * 2 + 2

    @property
    def max_output_len(self) -> int:
        # The output has the form "=wworrdd12<eos>".
        # Additionally, we require two more tokens for "=" and "<eos>".
        return self._max_len_query_digit * 2 + 2

    def get_random_string(self, rnd):
        s = "".join(
            self.alphabet[
                jax.random.randint(
                    rnd,
                    shape=(1, self._max_len_query_digit),
                    minval=0,
                    maxval=self.alphabet.shape[0],
                )
            ][0]
        )
        return s

    def generate_examples(
        self, num_examples: int, rnd_key: jax.random.PRNGKey
    ) -> Generator[Tuple[str, str], None, None]:
        for _ in range(num_examples):
            rnd_key, rnd_1 = jax.random.split(rnd_key)
            rnd_key, rnd_2 = jax.random.split(rnd_key)
            key = (self.get_random_string(rnd_1), self.get_random_string(rnd_2))
            inputs = f"{key[0]}+{key[1]}"
            s = ''
            for i in range(len(key[0]) + len(key[1])):
                if i % 2 == 0:
                    s += key[0][i // 2]
                else:
                    s += key[1][i // 2]
            outputs = f"={s}"
            yield (inputs, outputs)


class WordReverseTaskCT(CharacterTable):
    def __init__(self, max_len_query_digit: int = 3) -> None:
        self.alphabet = np.array([ch for ch in string.ascii_lowercase])
        chars = f"{string.ascii_lowercase}= "
        super().__init__(chars, max_len_query_digit)

    @property
    def max_input_len(self) -> int:
        # The input has the form "word<eos>", so the max input length is
        # the length of word plus one token for the EOS token.
        return self._max_len_query_digit + 1

    @property
    def max_output_len(self) -> int:
        # The output has the form "=work<eos>".
        # Additionally, we require two more tokens for "=" and "<eos>".
        return self._max_len_query_digit + 2

    def get_random_string(self, rnd):
        s = "".join(
            self.alphabet[
                jax.random.randint(
                    rnd,
                    shape=(1, self._max_len_query_digit),
                    minval=0,
                    maxval=self.alphabet.shape[0],
                )
            ][0]
        )
        return s

    def generate_examples(
        self, num_examples: int, rnd_key: jax.random.PRNGKey
    ) -> Generator[Tuple[str, str], None, None]:
        for _ in range(num_examples):
            rnd_key, rnd = jax.random.split(rnd_key)
            key = self.get_random_string(rnd)
            yek = key[::-1]
            inputs = key
            outputs = f"={yek}"
            yield (inputs, outputs)


class FloatAdditionTaskCT(CharacterTable):
    def __init__(self, max_len_query_digit: int = 3) -> None:
        chars = "0123456789+=. "
        super().__init__(chars, max_len_query_digit)


class ConsonantRemovingTaskCT(CharacterTable):
    pass


class LetterUppercasingMaskTaskCT(CharacterTable):
    pass


class SqrtTaskCT(CharacterTable):
    pass


class SinTaskCT(CharacterTable):
    pass


class MaskLetterRemovingCT(CharacterTable):
    pass


tasks = {
    "digit_addition": AdditionTaskCT,
    "word_addition": WordAdditionTaskCT,
    "word_diff_addition": WordDiffAdditionTaskCT,
    "word_reverse": WordReverseTaskCT,
    "float_addition": FloatAdditionTaskCT,
    "consonant_removing": ConsonantRemovingTaskCT,
    "letter_uppercase": LetterUppercasingMaskTaskCT,
    "sqrt": SqrtTaskCT,
    "sin": SinTaskCT,
}
