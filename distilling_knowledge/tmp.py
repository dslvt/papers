import jax
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import optax
import numpy as np
from flax.training import train_state, checkpoints, early_stopping
from flax.metrics import tensorboard

 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)),
    transforms.Lambda(lambda x: torch.flatten(x))
])

 
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


 
@jax.jit
def apply_model(state, data, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, data)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy

 
@jax.jit
def update_model(state, grads): 
    return state.apply_gradients(grads=grads)

 
def train_epoch(state, train_dt, rng):
    epoch_loss = []
    epoch_accuracy = []

    for batch_idx, (data, target) in enumerate(train_dt):
        data, target = data.numpy(), target.numpy()
        data, target = jnp.float32(data), jnp.float32(target)

        grads, loss, accuracy = apply_model(state, data, target)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)

    return state, train_loss, train_accuracy

 
def create_train_state(rng, config):
    mlp = MLP([1200, 1200, 10])
    params = mlp.init(rng, jnp.ones([1, 784]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=mlp.apply, params=params, tx=tx
    )

 
def train_and_evaluate(config, workdir):
    train = MNIST(root='data/', train=True, download=True, transform=transform)
    test = MNIST(root='data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test, **test_kwargs)

    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    early_stop = early_stopping.EarlyStopping(min_delta=1e-3, patience=2)

    for epoch in range(1, config['num_epoch'] + 1):
        rng, init_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_loader, rng)
        _, early_stop = early_stop.update(train_loss)
        test_dt, test_labels = next(iter(test_loader))
        _, test_loss, test_accuracy = apply_model(state, test_dt.numpy(), test_labels.numpy())
        print(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))
        
        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)
        checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=test_accuracy, prefix='mlp_1200', overwrite=True)

        if early_stop.should_stop:
            print('Met early stopping criteria, breaking...')
            break

    summary_writer.flush()
    return state
    

 
config = {'num_epoch': 10}
train_kwargs = {'batch_size': 8}
test_kwargs = {'batch_size': 120}
learning_rate = 1e-3
num_epoch = 1
CKPT_DIR = 'ckpts'
workdir = 'logs/'

 
state = train_and_evaluate(config, 'logs/')