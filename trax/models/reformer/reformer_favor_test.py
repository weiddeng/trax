# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Test for memory usage in Reformer models with FAVOR attention."""

from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import test_util  # pylint: disable=unused-import
from jax.config import config

import numpy as np

from trax import fastmath
from trax import layers as tl
from trax import shapes
from trax.models.reformer import reformer
from trax.supervised.training import _xprof_on_tpu


class ReformerFavorTest(absltest.TestCase):

  def test_reformer_favor_memory(self):

    model = reformer.ReformerLM(
        vocab_size=256,
        d_model=1024,
        d_ff=2048,
        n_layers=2,
        n_heads=8,
        max_len=12288,
        attention_type=tl.CausalFavor,
        mode='train',
    )
    x = np.ones((1, 12288)).astype(np.int32)
    weights, state = model.init(shapes.signature(x))

    @jax.jit
    def mock_training_step(x, weights, state, rng):
      def compute_mock_loss(weights):
        logits, new_state = model.pure_fn(x, weights, state, rng)
        loss = jnp.mean(logits[..., 0])
        return loss, (new_state, logits)
      gradients, (new_state, logits) = jax.grad(
          compute_mock_loss, has_aux=True)(weights)
      new_weights = fastmath.nested_map_multiarg(
          lambda w, g: w - 1e-4 * g, weights, gradients)
      return new_weights, new_state, logits

    # Pre-JIT one step.
    rng = jax.random.PRNGKey(0)
    weights, state, logits = mock_training_step(x, weights, state, rng)

    # Run xprof'd step.
    _xprof_on_tpu(3.0, 'step')
    weights, state, logits = mock_training_step(x, weights, state, rng)
    self.assertEqual(logits.shape, (1, 12288, 256))


if __name__ == '__main__':
  config.config_with_absl()
  absltest.main()
