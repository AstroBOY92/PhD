
import json
import os
from typing import NamedTuple, Tuple

import e3nn_jax as e3nn
from flax import serialization
import flax.linen as nn
import jax
from jax import eval_shape
from jax import random
from jax.tree_util import tree_map
from jax.core import ShapedArray
import jax.numpy as jnp
from jax_md import util
import jraph
from ml_collections import ConfigDict
import optax

from . import nequip


f32 = jnp.float32
i32 = jnp.int32

GraphsTuple = jraph.GraphsTuple

IrrepsArray = e3nn.IrrepsArray

NUM_ELEMENTS = 94

PyTree = util.PyTree


def model_from_config(cfg: ConfigDict) -> nn.Module:
  model_family = cfg.get('model_family', 'nequip')
  if model_family == 'nequip':
    return nequip.model_from_config(cfg)
  else:
    raise ValueError(f'Unrecognized model family: {model_family}')


def minimum_batch_size(cfg: ConfigDict) -> int:
  if not hasattr(cfg, 'train_batch_size'):
    return 1
  if isinstance(cfg.train_batch_size, int):
    return cfg.train_batch_size
  return min(cfg.train_batch_size)


class ScaleLROnPlateau(NamedTuple):
  step_size: jnp.ndarray
  minimum_loss: jnp.ndarray
  steps_without_reduction: jnp.ndarray
  max_steps_without_reduction: jnp.ndarray
  reduction_factor: jnp.ndarray


def scale_lr_on_plateau(
    initial_step_size: float,
    max_steps_without_reduction: int,
    reduction_factor: float,
) -> optax.GradientTransformation:
  def init_fn(params):
    del params
    return ScaleLROnPlateau(
        initial_step_size,
        jnp.inf,
        0,
        max_steps_without_reduction,
        reduction_factor,
    )

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(lambda g: g * state.step_size, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def optimizer(cfg: ConfigDict) -> optax.OptState:
  epoch_size = cfg.epoch_size if hasattr(cfg, 'epoch_size') else -1
  batch_size = minimum_batch_size(cfg)
  total_steps = cfg.epochs * (epoch_size // batch_size)
  warmup_steps = cfg.get('warmup_steps', 0)

  if cfg.schedule == 'constant':
    schedule = cfg.learning_rate
  elif cfg.schedule == 'linear_decay':
    schedule = optax.polynomial_schedule(cfg.learning_rate, 0.0, 1, total_steps)
  elif cfg.schedule == 'cosine_decay':
    schedule = optax.cosine_decay_schedule(cfg.learning_rate, total_steps)
  elif cfg.schedule == 'warmup_cosine_decay':
    schedule = optax.warmup_cosine_decay_schedule(
        1e-7, cfg.learning_rate, warmup_steps, total_steps
    )
  elif cfg.schedule == 'scale_on_plateau':
    max_plateau_steps = cfg.max_lr_plateau_epochs // cfg.epochs_per_eval
    return optax.chain(
        optax.scale_by_adam(),
        scale_lr_on_plateau(-cfg.learning_rate, max_plateau_steps, 0.8),
    )
  else:
    raise ValueError(f'Unknown learning rate schedule, "{cfg.schedule}".')

  if not hasattr(cfg, 'l2_regularization') or cfg.l2_regularization == 0.0:
    return optax.adam(schedule)

  return optax.adamw(schedule, weight_decay=cfg.l2_regularization)


def load_model(directory: str) -> Tuple[ConfigDict, nn.Module, PyTree]:
  with open(os.path.join(directory, 'config.json'), 'r') as f:
    c = json.loads(json.loads(f.read()))
    c = ConfigDict(c)

  # Now initialize the model and the optimizer functions.
  model = model_from_config(c)
  opt_init, _ = optimizer(c)

  graph = GraphsTuple(
      ShapedArray((1, NUM_ELEMENTS), f32),  # Nodes     (nodes, features)
      ShapedArray((1, 3), f32),  # dR        (edges, spatial)
      ShapedArray((1,), i32),  # senders   (edges,)
      ShapedArray((1,), i32),  # receivers (edges,)
      ShapedArray((1, 1), f32),  # globals   (graphs,)
      ShapedArray((1,), i32),  # n_node    (graphs,)
      ShapedArray((1,), i32),
  )  # n_edge    (graphs,)

  def init_opt_and_model(graph):
    key = random.PRNGKey(0)
    params = model.init(key, graph)
    state = opt_init(params)
    return params, state

  abstract_params, abstract_state = eval_shape(init_opt_and_model, graph)

  # Now that we have the structure, load the data using FLAX checkpointing.
  ckpt_data = (0, abstract_params, abstract_state)

  checkpoints = [c for c in os.listdir(directory) if 'checkpoint' in c]
  assert len(checkpoints) == 1

  checkpoint = os.path.join(directory, checkpoints[0])

  with open(checkpoint, 'rb') as f:
    ckpt = serialization.from_bytes(ckpt_data, f.read())

  params = tree_map(lambda x: x.astype(f32), ckpt[1])
  return c, model, params
