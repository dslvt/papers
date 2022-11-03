import sys
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from flax import linen as nn
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint
from jax import random


sys.path.append("..")
from utils import save_image

Array = Any

FLAGS = flags.FLAGS


flags.DEFINE_float(
    "learning_rate", default=1e-3, help=("The learning rate for the Adam optimizer.")
)

flags.DEFINE_integer("batch_size", default=128, help=("Batch size for training."))

flags.DEFINE_integer("num_epochs", default=30, help=("Number of training epochs."))

flags.DEFINE_integer("latents", default=20, help=("Number of latent variables."))


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x) -> Tuple[Array, Array]:
        x = nn.Dense(500, name="fc1")(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        return mean_x, logvar_x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z) -> Array:
        z = nn.Dense(500, name="fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(784, name="fc2")(z)
        return z


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


@jax.vmap
def kl_divergence(mean, logval):
    return -0.5 * jnp.sum(1 + logval - jnp.square(mean) - jnp.exp(logval))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return {"bce": bce_loss, "kld": kld_loss, "loss": bce_loss + kld_loss}


def model():
    return VAE(latents=FLAGS.latents)


@jax.jit
def train_step(state, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = model().apply({"params": params}, batch, z_rng)

        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval(params, images, z, z_rng):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate(
            [images[:8].reshape(-1, 28, 28, 1), recon_images[:8].reshape(-1, 28, 28, 1)]
        )
        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)

        print(comparison)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, model())({"params": params})


def prepare_image(x):
    x = tf.cast(x["image"], tf.float32)
    x = tf.reshape(x, (-1,))
    return x


def main(argv):
    del argv

    tf.config.experimental.set_visible_devices([], "GPU")

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    ds_builder = tfds.builder("binarized_mnist")
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(FLAGS.batch_size)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(10000)
    test_ds = np.array(list(test_ds)[0])
    test_ds = jax.device_put(test_ds)

    init_data = jnp.ones((FLAGS.batch_size, 784), jnp.float32)

    state = train_state.TrainState.create(
        apply_fn=model().apply,
        params=model().init(key, init_data, rng)["params"],
        tx=optax.adam(FLAGS.learning_rate),
    )

    rng, z_rng, eval_rng = random.split(rng, 3)
    z = random.normal(z_rng, (64, FLAGS.latents))

    steps_per_epoch = 50000 // FLAGS.batch_size

    for epoch in range(FLAGS.num_epochs):
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            rng, key = random.split(rng)
            state = train_step(state, batch, key)

        metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
        save_image(comparison, f"results/reconstraction_{epoch}.png", nrow=8)
        save_image(sample, f"results/sample_{epoch}.png", nrow=8)

        print(
            f"eval epoch: {epoch}, loss: {metrics['loss']:.4f}, BCE: {metrics['bce']:.4f}, KLD: {metrics['kld']:.4f}"
        )


if __name__ == "__main__":
    app.run(main)
