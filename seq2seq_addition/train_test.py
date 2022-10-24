import functools

from absl.testing import absltest
from flax.training import train_state
import jax
from jax import random
import numpy as np
import optax

import input_pipeline
import seq2seq_addition.train as train
import models

jax.config.parse_flags_with_absl()


def create_ctable(chars="0123456789+= "):
    return input_pipeline.CharacterTable(chars)


def create_training_state(ctable):
    model = models.Seq2seq(
        teacher_force=False,
        hidden_size=train.flags["hidden_size"],
        vocab_size=ctable.vocab_size,
    )
    params = train.get_initial_params(model, jax.random.PRNGKey(0), ctable)
    tx = optax.adam(train.flags["learning_rate"])
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


class TrainTest(absltest.TestCase):
    def test_character_table(self):
        ctable = create_ctable()
        text = "112+11"
        enc_text = ctable.encode(text)
        dec_text = ctable.decode(enc_text)

        self.assertEqual(text, dec_text.strip())

    def test_mask_sequence(self):
        np.testing.assert_equal(
            input_pipeline.mask_sequence(
                np.arange(1, 13).reshape((4, 3)), np.array([3, 2, 1, 0])
            ),
            np.array([[1, 2, 3], [4, 5, 0], [7, 0, 0], [0, 0, 0]]),
        )

    def test_get_sequence_lengths(self):
        oh_sequence_batch = jax.vmap(functools.partial(jax.nn.one_hot, num_classes=4))(
            np.array([[0, 1, 0], [1, 0, 2], [1, 2, 0], [1, 2, 3]])
        )
        np.testing.assert_equal(
            input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=0),
            np.array([1, 2, 3, 3], np.int32),
        )
        np.testing.assert_equal(
            input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=1),
            np.array([2, 1, 1, 1], np.int32),
        )
        np.testing.assert_equal(
            input_pipeline.get_sequence_lengths(oh_sequence_batch, eos_id=2),
            np.array([3, 3, 2, 2], np.int32),
        )

    def test_train_one_step(self):
        ctable = create_ctable()
        key = random.PRNGKey(0)
        batch = ctable.get_batch(128, key)

        state = create_training_state(ctable)
        _, training_metrics = train.train_step(state, batch, key, ctable.eos_id)

        self.assertLessEqual(training_metrics["loss"], 5)
        self.assertGreaterEqual(training_metrics["accuracy"], 0)

    def test_decode_batch(self):
        ctable = create_ctable()
        key = random.PRNGKey(0)
        batch = ctable.get_batch(5, key)
        state = create_training_state(ctable)
        train.decode_batch(state, batch, key, ctable)


if __name__ == "__main__":
    absltest.main()
