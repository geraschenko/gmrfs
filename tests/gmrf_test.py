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

"""Tests for gmrf.py."""

from absl.testing import absltest
from gmrfs import gmrf
from gmrfs import sparse
import jax
import numpy as np
import scipy.sparse
import scipy.stats

jax.config.update('jax_enable_x64', True)


def random_gaussian(dimension, rng):
  """A random gaussian."""
  mat = np.zeros((dimension, dimension))
  for i in range(dimension):
    # Push diagonal entries away from 0 to avoid tiny eigenvalues.
    mat[i, i] = rng.normal(loc=1)
  num_additional_entries = 2 * dimension
  for _ in range(num_additional_entries):
    i = rng.integers(1, dimension)
    j = rng.integers(i)
    mat[i, j] = rng.normal()
  mat = scipy.sparse.csc_matrix(mat)
  precision = sparse.SPDMatrix.from_scipy(mat @ mat.T)
  mean = rng.normal(size=precision.shape[1])
  return gmrf.Gaussian(precision, mean)


class GaussianTest(absltest.TestCase):

  def test_logpdf(self):
    rng = np.random.default_rng(0)
    gaussian = random_gaussian(dimension=10, rng=rng)
    scipy_gaussian = scipy.stats.multivariate_normal(
        mean=gaussian.mean,
        cov=np.linalg.inv(gaussian.precision.to_scipy().todense()),
    )
    num_test_points = 13
    x = rng.normal(size=(num_test_points, gaussian.mean.shape[0]))
    np.testing.assert_allclose(
        scipy_gaussian.logpdf(x), jax.vmap(gaussian.logpdf)(x)
    )

  def test_sample(self):
    # Draw samples and check that the mean and covariance match expectations.
    gaussian = random_gaussian(dimension=3, rng=np.random.default_rng(0))
    num_samples = 1000
    key = jax.random.PRNGKey(0)
    samples = jax.vmap(gaussian.sample)(jax.random.split(key, num_samples))
    estimated_mean = samples.mean(axis=0)
    np.testing.assert_almost_equal(
        gaussian.logpdf(estimated_mean),
        gaussian.logpdf(gaussian.mean),
        decimal=3,
    )
    centered_samples = samples - gaussian.mean
    estimated_cov = centered_samples.T @ centered_samples / len(samples)
    np.testing.assert_allclose(
        estimated_cov,
        np.linalg.inv(gaussian.precision.to_scipy().todense()),
        atol=3 / np.sqrt(num_samples),
    )


class RandomWalkTest(absltest.TestCase):

  def test_parameterization(self):
    # It's faster to draw more samples than to sample from larger walks.
    num_steps = 100
    num_samples = 1000
    rw = gmrf.RandomWalk(start_location=1.2, num_steps=num_steps)
    param = 0.43
    gaussian = rw(param)
    x = jax.vmap(gaussian.sample)(
        jax.random.split(jax.random.PRNGKey(0), num_samples)
    )
    steps = np.diff(x, axis=-1)
    expected_variance = np.exp(param)
    np.testing.assert_allclose(
        0.0, steps.mean(), atol=3 * np.sqrt(expected_variance / steps.size)
    )
    np.testing.assert_allclose(
        expected_variance,
        steps.var(),
        atol=3 * expected_variance / np.sqrt(steps.size),
    )


if __name__ == '__main__':
  absltest.main()
