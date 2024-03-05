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

"""JAX-compatible sparse positive definite symmetric matrices.

This library provides utilities for dealing with sparse positive definite
symmetric matrices with fixed sparsity structure. The main class is
SPDSparsityStructure, which delegates to sksparse to determine a sparse Cholesky
factorization and implements versions of all the methods we need for SPD
matrices (e.g. sparse cholesly factorization, triangular solves, diagonal
updates, log determinant) which operate on the underlying data vector of a
sparse SPD matrix with given sparsity structure.

SPDMatrix is a JAX-compatible class consisting of a (static)
SPDSparsityStructure together with the (dynamic) data vector of entries.

Important caveat: the sparse Cholesky factorization is implemented by simply
looping over the non-zero entries of the Cholesky triangle. So even though
methods using the SPDMatrix class are jittable, vmappable, differentiable, and
can run on accelerators, they're going to be slow. Replacing this loop with a
jax.lax.scan would be swell, but the naive way to do so loses the benefit of
sparsity.
"""

# We follow the naming convensions in sksparse.
# pylint:disable=invalid-name

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.sparse
import sksparse
import sksparse.cholmod

EdgeList = list[Tuple[Any, Any]]
UnaryOperation = Callable[[jnp.ndarray], jnp.ndarray]
BinaryOperation = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

NOT_POPULATED = -1


def data_index(mat: scipy.sparse.csc_matrix, i: int, j: int) -> int:
  """Index of mat[i, j] in mat.data, or NOT_POPULATED if mat[i, j] not set."""
  assert mat.has_sorted_indices, 'Indices are not sorted.'
  row_indices = mat.indices[mat.indptr[j] : mat.indptr[j + 1]]
  k = np.searchsorted(row_indices, i)
  if k < len(row_indices) and row_indices[k] == i:
    return mat.indptr[j] + k
  else:
    return NOT_POPULATED


def row_col_index(mat: scipy.sparse.csc_matrix, k: int) -> Tuple[int, int]:
  """Returns (i, j) so that mat[i, j] = mat.data[k]."""
  i = mat.indices[k]
  j = np.searchsorted(mat.indptr, k, side='right') - 1
  return i, j


def get_entries(data: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
  """data[indices], except an index of NOT_POPULATED produces 0.0."""
  return jnp.concatenate([data, jnp.array([0.0])])[indices]


def sparse_cholesky_fn(
    Q: scipy.sparse.csc_matrix, factor: Optional[sksparse.cholmod.Factor] = None
):
  """Returns sparse Cholesky factorization function.

  The returned method can be used for matrices with the same sparsity structure
  as Q.

  Args:
    Q: a sparse symmetric positive definite matrix.
    factor: a sksparse.cholmod.Factor representing the sparse Cholesky
      factorization of Q.

  Returns:
    A function which, given a specification of the entries Q.data, returns the
    values L.data of the Cholesky factorization.
  """
  if factor is None:
    factor = sksparse.cholmod.cholesky(Q)
  L = factor.L()
  P = factor.P()

  # Determine the compressed sparse *row* representation of the Cholesky
  # triangle so that we can efficiently compute dot products of rows as we fill
  # in the columns of L.
  L_row = scipy.sparse.csr_matrix(L)

  # row_overlaps[(i, j)] is a pair of index arrays (ind1, ind2) recording the
  # overlap of row i with row j. It has the property that
  # L_row.data[ind1] @ L_row.data[ind2] + L[i, j] * L[j, j]
  # computes the dot product of row i and row j.
  # We compute the Cholesky factorization by setting this equal to Q[i, j] and
  # solving for L[j, i]. We only populate the row overlaps that will be required
  # to compute the Cholesky factorization.
  row_overlaps = {}
  for j in range(Q.shape[1]):
    row1_start, row1_end = L_row.indptr[j], L_row.indptr[j + 1]
    columns1 = L_row.indices[row1_start:row1_end]
    for i in L.indices[L.indptr[j] : L.indptr[j + 1]]:
      row2_start, row2_end = L_row.indptr[i], L_row.indptr[i + 1] - 1
      columns2 = L_row.indices[row2_start:row2_end]
      _, indices1, indices2 = np.intersect1d(
          columns1, columns2, assume_unique=True, return_indices=True
      )
      row_overlaps[(i, j)] = (row1_start + indices1, row2_start + indices2)

  # Q_indices[k] is the index of the element of Q.data which is in the same row
  # and column as L.data[k] is within L.
  Q_indices = []
  for k in range(len(L.data)):
    i, j = row_col_index(L, k)
    Q_indices.append(data_index(Q, P[i], P[j]))
  Q_indices = jnp.array(Q_indices)

  def cholesky_factor(Q_data: jnp.ndarray) -> jnp.ndarray:
    Q_entries = get_entries(Q_data, Q_indices)
    L_data = jnp.zeros_like(L.data)
    L_row_data = jnp.zeros_like(L.data)
    # num_populated[i] tracks how many entries of row i been populated for
    # purposes of determining which entry of L_row_data to set.
    num_populated = [0 for _ in range(Q.shape[0])]
    for j in range(Q.shape[1]):
      # Compute the entries of L in column j.
      col_start, col_end = L.indptr[j], L.indptr[j + 1]
      row_start, row_end = L_row.indptr[j], L_row.indptr[j + 1]

      diag_entry = jnp.sqrt(
          Q_entries[col_start]
          - L_row_data[row_start : row_end - 1]
          @ L_row_data[row_start : row_end - 1]
      )

      col_entries = [diag_entry]
      row_indices = [row_end - 1]  # The diagonal entry is the last in its row.
      for k in range(col_start + 1, col_end):
        i = L.indices[k]
        # Compute the entry in row i, column j
        ind1, ind2 = row_overlaps[(i, j)]
        row_dot_other_row = L_row_data[ind1] @ L_row_data[ind2]
        entry = (Q_entries[k] - row_dot_other_row) / diag_entry
        col_entries.append(entry)
        other_row_start = L_row.indptr[i]
        row_indices.append(other_row_start + num_populated[i])
        num_populated[i] += 1

      L_data = L_data.at[col_start:col_end].set(col_entries)
      L_row_data = L_row_data.at[np.array(row_indices)].set(col_entries)
    return L_data, L_row_data  # pytype: disable=bad-return-type  # jax-ndarray

  return cholesky_factor


def is_lower_triangular(L: scipy.sparse.csc_matrix) -> bool:
  # All indices in the i-th column must be at least i.
  return all(
      (L.indices[L.indptr[i] : L.indptr[i + 1]] >= i).all()
      for i in range(L.shape[1])
  )


def is_upper_triangular(U: scipy.sparse.csc_matrix) -> bool:
  # All indices in the i-th column must be at most i.
  return all(
      (U.indices[U.indptr[i] : U.indptr[i + 1]] <= i).all()
      for i in range(U.shape[1])
  )


def tril_solver(
    L: scipy.sparse.csc_matrix, L_T: Optional[scipy.sparse.csc_matrix] = None
) -> BinaryOperation:
  """Returns a method that solves sparse lower triangular linear systems."""
  assert isinstance(L, scipy.sparse.csc_matrix)
  assert (
      L.shape[0] == L.shape[1]
  ), f'Only square (and full rank) matrices are supported. {L.shape=}'
  assert is_lower_triangular(L), 'L must be lower triangular.'
  assert all(
      [L.indices[L.indptr[i]] == i for i in range(L.shape[0])]
  ), 'L must have populated diagonal entries'

  if L_T is None:
    L_T = scipy.sparse.csc_matrix(L.T)
  else:
    assert isinstance(L_T, scipy.sparse.csc_matrix)

  def solve(L_T_data, b):
    x = jnp.zeros(L.shape[0])
    for i in range(L.shape[0]):
      row_start, row_end = L_T.indptr[i], L_T.indptr[i + 1]
      indices = L_T.indices[row_start : row_end - 1]
      entry = (
          b[i] - L_T_data[row_start : row_end - 1] @ x[indices]
      ) / L_T_data[row_end - 1]
      x = x.at[i].set(entry)
    return x

  return solve


def triu_solver(
    U: scipy.sparse.csc_matrix, U_T: Optional[scipy.sparse.csc_matrix] = None
) -> BinaryOperation:
  """Returns a method that solves sparse upper triangular linear systems."""
  assert isinstance(U, scipy.sparse.csc_matrix)
  assert (
      U.shape[0] == U.shape[1]
  ), f'Only square (and full rank) matrices are supported. {U.shape=}'
  assert is_upper_triangular(U), 'U must be upper triangular.'
  assert all(
      [U.indices[U.indptr[i + 1] - 1] == i for i in range(U.shape[0])]
  ), 'U must have populated diagonal entries'

  if U_T is None:
    U_T = scipy.sparse.csc_matrix(U.T)
  else:
    assert isinstance(U_T, scipy.sparse.csc_matrix)

  def solve(U_T_data, b):
    x = jnp.zeros(U.shape[0])
    for i in reversed(range(U.shape[0])):
      row_start, row_end = U_T.indptr[i], U_T.indptr[i + 1]
      indices = U_T.indices[row_start + 1 : row_end]
      entry = (
          b[i] - U_T_data[row_start + 1 : row_end] @ x[indices]
      ) / U_T_data[row_start]
      x = x.at[i].set(entry)
    return x

  return solve


def add_fn(
    Q: scipy.sparse.csc_matrix,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Returns a method that adds a diagonal matrix diag(x) to Q."""
  assert isinstance(Q, scipy.sparse.csc_matrix)
  assert (
      Q.shape[0] == Q.shape[1]
  ), f'Only square (and full rank) matrices are supported. {Q.shape=}'

  Q_diag_inds = jnp.array([data_index(Q, i, i) for i in range(Q.shape[0])])
  assert -1 not in Q_diag_inds, 'Diagonal entries of Q must be populated.'

  def Q_add(Q_data, x):
    Q_data = Q_data.at[Q_diag_inds].set(Q_data[Q_diag_inds] + x)
    return Q_data

  return Q_add


def multiply_fn(
    Q: scipy.sparse.csc_matrix, Q_T: Optional[scipy.sparse.csc_matrix] = None
) -> BinaryOperation:
  """Returns a method that computes Q @ x, where x is a dense vector."""
  assert isinstance(Q, scipy.sparse.csc_matrix)
  assert (
      Q.shape[0] == Q.shape[1]
  ), f'Only square (and full rank) matrices are supported. {Q.shape=}'
  if Q_T is None:
    Q_T = scipy.sparse.csc_matrix(Q.T)

  def Q_multiply(Q_T_data, x):
    y = []
    for i in range(Q.shape[0]):
      row_start, row_end = Q_T.indptr[i], Q_T.indptr[i + 1]
      indices = Q_T.indices[row_start:row_end]
      y.append(Q_T_data[row_start:row_end] @ x[indices])
    return jnp.array(y)

  return Q_multiply


def logdet_fn(L: scipy.sparse.csc_matrix) -> UnaryOperation:
  """Returns a method that computes the log determinant of L."""
  assert isinstance(L, scipy.sparse.csc_matrix)
  assert is_lower_triangular(L) or is_upper_triangular(
      L
  ), 'L must be triangular'
  assert (
      L.shape[0] == L.shape[1]
  ), f'Only square (and full rank) matrices are supported. {L.shape=}'

  L_diag_inds = jnp.array([data_index(L, i, i) for i in range(L.shape[0])])

  def L_log_det(L_data):
    L_diag = L_data[L_diag_inds]
    return jnp.log(L_diag).sum()

  return L_log_det


def node_names_and_integer_edges(edges: EdgeList) -> tuple[list[Any], EdgeList]:
  node_names = sorted(set.union(*[set(e) for e in edges]))
  int_edges = [(node_names.index(i), node_names.index(j)) for i, j in edges]
  return node_names, int_edges


def adjacency_matrix(edges: EdgeList) -> scipy.sparse.csc_matrix:
  nodes, edges = node_names_and_integer_edges(edges)
  row_indices, col_indices = zip(*edges)
  dim = len(nodes)
  directed_adjacency_matrix = scipy.sparse.csc_matrix(
      (np.ones_like(row_indices), (row_indices, col_indices)), shape=(dim, dim)
  )
  return directed_adjacency_matrix + directed_adjacency_matrix.T


def edge_list(mat: scipy.sparse.spmatrix) -> EdgeList:
  """An edge list matching the sparsity structure of `mat`."""
  mat = mat.tocoo()
  return list(set(tuple(sorted((r, c))) for r, c in zip(mat.row, mat.col)))


class SPDSparsityStructure:
  """Deals with data vectors related to a given sparsity structure."""

  edges: EdgeList
  # Runtime asserts can be turned off by setting this member to False. This is
  # required to avoid ConcretizationTypeError when you apply jax.vmap.
  do_asserts: bool

  # Don't set these at initialization; they will be overwritten. These are just
  # here to make pytype happy.
  node_names: Optional[list[Any]] = None
  int_edges: Optional[EdgeList] = None
  dim: Optional[int] = None
  Q: Optional[scipy.sparse.csc_matrix] = None
  data_size: Optional[int] = None
  L: Optional[scipy.sparse.csc_matrix] = None
  L_T: Optional[scipy.sparse.csc_matrix] = None
  P: Optional[np.ndarray] = None
  Pinv: Optional[np.ndarray] = None
  cholesky_factor: Optional[UnaryOperation] = None
  solve_L: Optional[BinaryOperation] = None
  solve_LT: Optional[BinaryOperation] = None
  multiply_Q: Optional[BinaryOperation] = None
  add_diag_Q: Optional[BinaryOperation] = None
  logdet_L: Optional[UnaryOperation] = None

  def __init__(self, edges: EdgeList):
    self.edges = edges
    self.do_asserts = True

    self.node_names, self.int_edges = node_names_and_integer_edges(self.edges)
    self.dim = len(self.node_names)
    # Note: the diagonal is multiplied by self.dim to ensure positive
    # definiteness so that cholesky factorization doesn't crash.
    self.Q = adjacency_matrix(self.int_edges) + self.dim * scipy.sparse.eye(
        self.dim
    )
    self.data_size = len(self.Q.data)
    Q_coo = self.Q.tocoo()
    self.transpose_indices = np.array(
        [data_index(self.Q, row, col) for row, col in zip(Q_coo.col, Q_coo.row)]
    )

    factor = sksparse.cholmod.cholesky(self.Q)
    self.L = factor.L()
    self.L_T = scipy.sparse.csc_matrix(self.L.T)
    self.P = factor.P()
    self.Pinv = np.argsort(self.P)
    self.cholesky_factor = sparse_cholesky_fn(self.Q, factor=factor)
    self.solve_L = tril_solver(L=self.L, L_T=self.L_T)
    self.solve_LT = triu_solver(U=self.L_T, U_T=self.L)
    self.multiply_Q = multiply_fn(self.Q, Q_T=self.Q)
    self.add_diag_Q = add_fn(self.Q)
    self.logdet_L = logdet_fn(self.L)

  # WARNING: this method is likely to disappear or change behavior!  I'm not
  # sure what the best way to handle this is. The sparsity structure gets passed
  # around a lot (e.g. it's inherited by all SPDMatrices output by a given
  # GMRF), so it's worth making its methods fast.
  def jit_me(self) -> None:
    for method in [
        'cholesky_factor',
        'solve_L',
        'solve_LT',
        'multiply_Q',
        'add_diag_Q',
        'logdet_L',
    ]:
      setattr(self, method, jax.jit(getattr(self, method)))

  def __repr__(self) -> str:
    return f'SPDSparsityStructure(edges={self.edges})'

  def transpose(self, data: jnp.ndarray) -> jnp.ndarray:
    return data[self.transpose_indices]

  def data_from_scipy(self, mat: scipy.sparse.spmatrix) -> jnp.ndarray:
    """Data vector for `mat`, aligned with Q.data."""
    data = np.zeros_like(self.Q.data)
    mat = mat.tocoo()
    for value, row, col in zip(mat.data, mat.row, mat.col):
      idx = data_index(self.Q, row, col)
      if idx == NOT_POPULATED:
        raise ValueError(
            f'Entry ({row}, {col}) is populated in mat, but edge '
            'is not in the graph.'
        )
      data[idx] = value
    return jnp.array(data)

  def data_to_scipy(self, data: jnp.ndarray) -> scipy.sparse.csc_matrix:
    if self.do_asserts:
      assert (
          len(data) == self.data_size
      ), f'Wrong data size ({len(data)} vs {self.data_size})'
    indptr = self.Q.indptr
    indices = self.Q.indices
    dim = self.dim
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(dim, dim))

  def to_spd_matrix(self, data: jnp.ndarray) -> SPDMatrix:
    spd_matrix = SPDMatrix(self, data)
    if self.do_asserts:
      assert (
          len(data) == self.data_size
      ), f'Wrong data size ({len(data)} vs {self.data_size})'
      assert jnp.all(
          self.transpose(data) == data
      ), 'Data vector does not correspond to a symmetric matrix.'
      assert not jnp.isnan(
          spd_matrix.logdet
      ), 'Data vector does not correspond to a positive definite matrix.'
    return spd_matrix


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SPDMatrix:
  """A sparse positive definite matrix."""

  sparsity_structure: SPDSparsityStructure
  data: jnp.ndarray

  def __matmul__(self, vec: jnp.ndarray) -> jnp.ndarray:
    return self.sparsity_structure.multiply_Q(self.data, vec)

  def add_diag(self, vec: jnp.ndarray) -> SPDMatrix:
    return SPDMatrix(
        self.sparsity_structure,
        self.sparsity_structure.add_diag_Q(self.data, vec),
    )

  def to_scipy(self) -> scipy.sparse.csc_matrix:
    return self.sparsity_structure.data_to_scipy(self.data)

  @classmethod
  def from_scipy(cls, mat: scipy.sparse.spmatrix) -> SPDMatrix:
    mat = mat.tocoo()
    edges = edge_list(mat)
    sparsity_structure = SPDSparsityStructure(edges)
    data = sparsity_structure.data_from_scipy(mat)
    return SPDMatrix(sparsity_structure, data)

  def solve_LT(self, epsilon: jnp.ndarray) -> jnp.ndarray:
    """Solves L.T @ x = epsilon, where Q = L @ L.T."""
    L_data, _ = self._cholesky_factorization
    x = self.sparsity_structure.solve_LT(L_data, epsilon)
    return x[self.sparsity_structure.Pinv]

  def solve(self, b: jnp.ndarray) -> jnp.ndarray:
    """Solves for Q @ x = b."""
    L_data, LT_data = self._cholesky_factorization
    v = self.sparsity_structure.solve_L(LT_data, b[self.sparsity_structure.P])
    x = self.sparsity_structure.solve_LT(L_data, v)
    return x[self.sparsity_structure.Pinv]

  @functools.cached_property
  def _cholesky_factorization(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return self.sparsity_structure.cholesky_factor(self.data)

  @functools.cached_property
  def logdet(self) -> jnp.ndarray:
    L_data, _ = self._cholesky_factorization
    return 2.0 * self.sparsity_structure.logdet_L(L_data)

  @property
  def shape(self) -> Tuple[int, int]:
    return (self.sparsity_structure.dim, self.sparsity_structure.dim)

  # Methods below for JAX pytree compatibility.
  def tree_flatten(self):
    dynamic_values = (self.data,)
    static_values = (self.sparsity_structure,)
    return dynamic_values, static_values

  @classmethod
  def tree_unflatten(cls, static_values, dynamic_values):
    return cls(*static_values, *dynamic_values)
