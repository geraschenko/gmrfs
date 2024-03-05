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

"""Gaussian Markov Random Field models.

These are parameterized families of multivariate gaussians with sparse precision
matrices.
"""

# We follow the naming convensions in sksparse.
# pylint:disable=invalid-name

from __future__ import annotations

import dataclasses
import functools

from gmrfs import sparse
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy


def astuple_shallow(obj):
  """Shallow version of dataclasses.astuple."""
  return tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Gaussian:
  """Represents a multivariate normal with sparse precision matrix."""

  precision: sparse.SPDMatrix
  mean: jnp.ndarray

  @functools.cached_property
  def information_vector(self):
    return self.precision @ self.mean

  def logpdf(self, x: jnp.ndarray) -> jnp.ndarray:
    offset = x - self.mean
    return (
        -0.5 * offset @ (self.precision @ offset)
        + 0.5 * self.precision.logdet
        - self.precision.sparsity_structure.dim * jnp.log(2 * jnp.pi) / 2
    )

  def sample(self, key: jax.Array) -> jnp.ndarray:
    """Draws a single sample from this gaussian."""
    epsilon = jax.random.normal(key, shape=self.mean.shape)
    x = self.precision.solve_LT(epsilon)
    return x + self.mean

  # Methods below for JAX pytree compatibility.
  def tree_flatten(self):
    return astuple_shallow(self), None

  @classmethod
  def tree_unflatten(cls, _, fields):
    return cls(*fields)


class Model:
  """A parameterized family of gaussians with sparse precision matrices."""

  def init_params(self):
    """Returns something of the right shape to pass to __call__."""
    raise NotImplementedError

  def __call__(self, params) -> Gaussian:
    """Returns the precision matrix and mean for given parameter values."""
    raise NotImplementedError


class RandomWalk(Model):
  """A random walk with given start location and a parameter for step size."""

  def __init__(self, start_location, num_steps):
    self.mean = start_location * jnp.ones(num_steps)
    edges = [(i, i + 1) for i in range(num_steps - 1)]
    self.sparsity_structure = sparse.SPDSparsityStructure(edges)
    precision = scipy.sparse.diags(
        [2] * (num_steps - 1) + [1]
    ) - sparse.adjacency_matrix(edges)
    self.precision_data = self.sparsity_structure.data_from_scipy(precision)

  def init_params(self) -> jnp.ndarray:
    """θ, where the steps have mean 0 and variance exp(θ)."""
    return jnp.array([0.0])

  def __call__(self, params) -> Gaussian:
    precision = self.sparsity_structure.to_spd_matrix(
        self.precision_data / jnp.exp(params)
    )
    return Gaussian(precision, self.mean)


class IntrinsicConditionalAutoregressive(Model):
  """An ICAR model with mean determined by a linear function of covariates."""

  def __init__(
      self, graph: nx.Graph, covariates: np.ndarray, regularization: float = 0.1
  ):
    self.covariates = covariates
    self.sparsity_structure = sparse.SPDSparsityStructure(list(graph.edges))
    W = self.sparsity_structure.data_from_scipy(
        sparse.adjacency_matrix(graph.edges)
    )
    D = self.sparsity_structure.data_from_scipy(
        scipy.sparse.diags(
            [regularization + graph.degree[i] for i in graph.nodes]
        )
    )
    self.precision_data = D - W

  def init_params(self):
    return (1.0, jnp.ones(self.covariates.shape[-1]))

  def __call__(self, params) -> Gaussian:
    tau, beta = params
    Q = self.sparsity_structure.to_spd_matrix(tau * self.precision_data)
    mu = self.covariates @ beta
    return Gaussian(Q, mu)
