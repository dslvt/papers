from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", default=".", help="Where to store log output.")
flags.DEFINE_string("model_name", default="vgg16", help="Model name")


def main(_):
    _ = train_and_evaluate(FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
