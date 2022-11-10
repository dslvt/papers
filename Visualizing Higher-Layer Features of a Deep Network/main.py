import os
import sys
from functools import partial

import flax.linen as nn
import flaxmodels as fm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from absl import app, flags, logging
from flax.core.frozen_dict import freeze
from jax import random
from optax import adamw

sys.path.append("..")
from utils import str_tree

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", default=".", help="Where to store log output.")
flags.DEFINE_integer("random_key", default=0, help="initial random key")
flags.DEFINE_float("learning_rate", default=0.01, help="learning rate")
flags.DEFINE_integer("steps", default=100, help="number of steps")
flags.DEFINE_integer("log_step", default=5, help="print loss in each log_step")
flags.DEFINE_integer(
    "image_save_step", default=10, help="save image in each image_save_step"
)
flags.DEFINE_string("layer_name", default="", help="name of layer")
flags.DEFINE_integer("filter_num", default=0, help="filter number")


def save_image(img, path, show=True):
    """Displays and saves the processsed image from the
    given layer/filter number.
    Arguments:
        - image (np.ndarray)
        - path (string) save path
    """
    plt.figure(figsize=[2, 2])
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    plt.savefig(path, dpi=300)
    if show:
        plt.show()


class ImageLayer(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("image", jax.nn.initializers.constant(1), (1, 224, 224, 3))
        return jax.lax.mul(inputs, kernel)


class ModelWrapper(nn.Module):
    backbone: nn.Module

    @nn.compact
    def __call__(self, x):
        x = ImageLayer()(x)

        x = self.backbone(x, train=False)
        return x


def train(model, params, img, filter_num, layer_name):
    @jax.jit
    def train_step(params, img):
        def loss_fn(params):
            act = model.apply(params, img)
            if "fc" in layer_name:
                act = act[layer_name][:, int(filter_num)]
            else:
                act = act[layer_name][0, :, :, int(filter_num)]

            loss = -jnp.mean(act)
            return loss

        loss_val, grad_val = jax.value_and_grad(loss_fn)(params)

        return loss_val, grad_val["params"]["ImageLayer_0"]["image"]

    return train_step(params, img)


def main(_):
    key = random.PRNGKey(FLAGS.random_key)
    rng_image = random.randint(key, shape=(1, 224, 224, 3), minval=83, maxval=171)
    rng_image = rng_image.astype(jnp.float32)
    rng_image = rng_image / 255

    model = ModelWrapper(fm.VGG19(output="activations", pretrained="imagenet"))
    vgg19 = fm.VGG19(output="activations", pretrained="imagenet")
    params = vgg19.init(key, rng_image, train=False)

    variables = model.init(jax.random.PRNGKey(1), rng_image)
    variables = variables.unfreeze()
    variables["params"]["backbone"] = params["params"]
    variables = freeze(variables)

    logging.info(str_tree(variables, 3))

    output_path = f"output/{FLAGS.model_name}/{FLAGS.layer_name}/{FLAGS.filter_num}"
    os.makedirs(output_path, exist_ok=True)

    optimizer = adamw(FLAGS.learning_rate, weight_decay=1e-6)
    opt_state = optimizer.init({"params": rng_image})

    for step in range(1, FLAGS.steps + 1):
        loss, grad = train(
            model,
            variables,
            rng_image,
            FLAGS.filter_num,
            FLAGS.layer_name,
        )
        updates, opt_state = optimizer.update(
            {"params": grad}, opt_state, {"params": rng_image}
        )
        rng_image = optax.apply_updates({"params": rng_image}, updates)
        rng_image = rng_image["params"]

        if step % FLAGS.log_step == 0:
            logging.info(f"step: {step} loss: {loss}")

        if step % FLAGS.image_save_step == 0:
            r_image = rng_image.squeeze(0)
            save_image(r_image, f"{output_path}/{step}.png", False)


if __name__ == "__main__":
    app.run(main)
