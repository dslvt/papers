import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

Array = Any
PRNGKey = Any


class EncoderLSTM(nn.Module):
    eos_id: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: Tuple[Array, Array], x: Array
    ) -> Tuple[Tuple[Array, Array], Array]:
        lstm_state, is_eos = carry
        new_lstm_state, y = nn.LSTMCell()(lstm_state, x)

        def select_carried_state(new_state, old_state):
            return jnp.where(is_eos[:, np.newaxis], old_state, new_state)

        carried_lstm_state = tuple(
            select_carried_state(*s) for s in zip(new_lstm_state, lstm_state)
        )
        is_eos = jnp.logical_or(is_eos, x[:, self.eos_id])
        return (carried_lstm_state, is_eos), y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )


class Encoder(nn.Module):
    hidden_size: int
    eos_id: int

    @nn.compact
    def __call__(self, inputs: Array):  # TODO: write output typing
        # inputs.shape = (batch_size, seq_length, vocab_size).
        batch_size = inputs.shape[0]
        lstm = EncoderLSTM(name="encoder_lstm", eos_id=self.eos_id)
        init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)
        init_is_eos = jnp.zeros(batch_size, dtype=bool)
        init_carry = (init_lstm_state, init_is_eos)
        (final_state, _), _ = lstm(init_carry, inputs)
        return final_state


class DecoderLSTM(nn.Module):
    teacher_force: bool
    vocab_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False, "lstm": True},
    )
    @nn.compact
    def __call__(self, carry: Tuple[Array, Array], x: Array) -> Array:
        lstm_state, last_prediction = carry
        if not self.teacher_force:
            x = last_prediction
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        logits = nn.Dense(features=self.vocab_size)(y)

        categorical_rng = self.make_rng("lstm")
        predicted_token = jax.random.categorical(categorical_rng, logits)
        prediction = jax.nn.one_hot(predicted_token, self.vocab_size, dtype=jnp.float32)

        return (lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
    init_state: Tuple[Any]
    teacher_force: bool
    vocab_size: int

    @nn.compact
    def __call__(self, inputs: Array) -> Tuple[Array, Array]:
        lstm = DecoderLSTM(teacher_force=self.teacher_force, vocab_size=self.vocab_size)
        init_carry = (self.init_state, inputs[:, 0])
        _, (logits, prediction) = lstm(init_carry, inputs)
        return logits, prediction


class Seq2seq(nn.Module):
    teacher_force: bool
    hidden_size: int
    vocab_size: int
    eos_id: int = 1

    @nn.compact
    def __call__(
        self, encoder_inputs: Array, decoder_inputs: Array
    ) -> Tuple[Array, Array]:
        init_decoder_state = Encoder(hidden_size=self.hidden_size, eos_id=self.eos_id)(
            encoder_inputs
        )

        logits, predictions = Decoder(
            init_state=init_decoder_state,
            teacher_force=self.teacher_force,
            vocab_size=self.vocab_size,
        )(decoder_inputs[:, :-1])

        return logits, predictions
