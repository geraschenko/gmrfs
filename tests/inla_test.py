# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for inla.py."""

from absl.testing import absltest
from gmrfs import gmrf
from gmrfs import inla
import jax
import numpy as np


class InlaTest(absltest.TestCase):

  def test_random_walk_fit(self):
    num_steps = 31  # An unusual number to catch shape bugs.
    num_samples = 1003  # An unusual number to catch shape bugs.
    rw = gmrf.RandomWalk(start_location=1.2, num_steps=num_steps)
    true_param = 0.43
    gaussian = rw(true_param)
    sample_key, measure_key = jax.random.split(jax.random.PRNGKey(0))
    x = jax.vmap(gaussian.sample)(jax.random.split(sample_key, num_samples))

    def measurement_log_prob_fn(x, y):
      return jax.scipy.stats.norm.logpdf(y, loc=x)

    y = x + jax.random.normal(measure_key, shape=x.shape)

    inla_model = inla.Inla(rw, measurement_log_prob_fn)

    params = np.arange(100) / 100.0
    # Set do_asserts to False to avoid vmap complaining.
    inla_model.gmrf_model.sparsity_structure.do_asserts = False
    likelihood = jax.vmap(
        jax.vmap(inla_model.log_marginal_posterior, in_axes=(0, None)),
        in_axes=(None, 0),
    )(params, y)
    # `total_likelihood` is the sum of likelihoods across all samples.
    total_likelihood = likelihood.sum(axis=0)

    # Assert that the true value close to the most likely. Increasing num_steps
    # or num_samples makes this sharper, but makes the test slow and memory
    # hungry.
    self.assertLess(
        np.abs(true_param - params[np.argmax(total_likelihood)]), 0.05
    )


if __name__ == '__main__':
  absltest.main()
