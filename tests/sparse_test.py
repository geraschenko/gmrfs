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

"""Tests for sparse."""

# Disable linting of invalid names. Names starting with "Q" or "L" (for
# variables related to the precisions matrix and the Cholesky triangle) help
# readability.
# pylint:disable=invalid-name

from absl.testing import absltest
from gmrfs import sparse
import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse
import sksparse.cholmod

# numpy uses float64 by default and jax uses float32 by default. In this file,
# we're just testing correctness of the algorithms, so use higher precision.
config.update('jax_enable_x64', True)


def csc_debug_str(i, j, k, mat):
  """A function that may come in handy debugging any test breakages."""
  return (
      f'{i=}, {j=}, {k=}\n{mat.indptr=}\n{mat.indices=}\n{mat.data=}\n'
      f'{mat.todense()}'
  )


def random_sparse_square_matrix(n, num_entries, rng):
  """A random sparse square."""
  mat = np.zeros((n, n))
  for _ in range(num_entries):
    i = rng.integers(n)
    j = rng.integers(n)
    mat[i, j] = rng.normal()
  return scipy.sparse.csc_matrix(mat)


def random_sparse_L(n, num_entries, rng):
  """A random sparse lower triangular matrix with nonzero diagonal."""
  mat = np.zeros((n, n))
  for i in range(n):
    mat[i, i] = rng.normal()
  for _ in range(num_entries - n):
    i = rng.integers(1, n)
    j = rng.integers(i)
    mat[i, j] = rng.normal()
  return scipy.sparse.csc_matrix(mat)


class SparseTest(absltest.TestCase):

  def test_csc_indexing(self):
    n = 7
    rng = np.random.default_rng(0)
    mat = random_sparse_square_matrix(n=n, num_entries=10, rng=rng)

    # Test that row_col_index values are correct.
    for k in range(len(mat.data)):
      i, j = sparse.row_col_index(mat, k)
      self.assertEqual(mat[i, j], mat.data[k])

    # Test that data_index values are correct.
    for i in range(n):
      for j in range(n):
        k = sparse.data_index(mat, i, j)
        if mat[i, j] == 0.0:
          self.assertEqual(k, sparse.NOT_POPULATED)
        else:
          self.assertEqual(mat[i, j], mat.data[k])

    # Test that get_entries behaves correctly.
    indices = np.array(
        [sparse.data_index(mat, i, j) for i in range(n) for j in range(n)]
    )
    entries = sparse.get_entries(mat.data, indices)
    # We only assert *almost* equal because np uses float64 by default and jnp
    # uses float32 by default.
    np.testing.assert_almost_equal(np.array(mat.todense()).ravel(), entries)

  def test_cholesky(self):
    # Pick a random symmetric positive definite matrix.
    rng = np.random.default_rng(0)
    mat = random_sparse_L(n=7, num_entries=10, rng=rng)
    Q = mat @ mat.T

    factor = sksparse.cholmod.cholesky(Q)
    L = factor.L()

    # This really just confirms our understanding of the meaning of P rather
    # than testing any of our code.
    P = factor.P()
    np.testing.assert_almost_equal((L @ L.T).todense(), Q[P, :][:, P].todense())

    # Test that our sparse factorization produces the same results as sksparse.
    cholesky_factor = sparse.sparse_cholesky_fn(Q, factor)
    np.testing.assert_almost_equal(L.data, cholesky_factor(Q.data)[0])

    # Test that cholesky_factor is jittable and differentiable.
    jax.jit(cholesky_factor)(Q.data)
    jax.jacobian(cholesky_factor)(Q.data)

    # Test that cholesky_factor is differentiable by computing its Jacobian.
    jax.jacfwd(cholesky_factor)(Q.data)

  def test_tril_solver(self):
    rng = np.random.default_rng(0)
    L = random_sparse_L(7, num_entries=10, rng=rng)
    b = rng.normal(size=L.shape[0])
    solver = sparse.tril_solver(L)

    # Change all the entries to check that we're not accidentally leaking L's
    # original data.
    other_L = L.copy()
    other_L.data = rng.normal(size=other_L.data.shape[0])
    other_L_T = scipy.sparse.csc_matrix(other_L.T)
    x = solver(other_L_T.data, b)

    np.testing.assert_allclose(other_L @ x, b)

    # Test that solver is jittable and differentiable.
    jax.jit(solver)(other_L_T.data, b)
    jax.jacobian(solver)(other_L_T.data, b)

    # Test that cholesky_factor is differentiable by computing its Jacobian.
    jax.jacfwd(solver)(other_L_T.data, b)

  def test_triu_solver(self):
    rng = np.random.default_rng(0)
    U = scipy.sparse.csc_matrix(random_sparse_L(7, num_entries=10, rng=rng).T)
    b = rng.normal(size=U.shape[0])
    solver = sparse.triu_solver(U)

    # Change all the entries to check that we're not accidentally leaking U's
    # original data.
    other_U = U.copy()
    other_U.data = rng.normal(size=other_U.data.shape[0])
    other_U_T = scipy.sparse.csc_matrix(other_U.T)
    x = solver(other_U_T.data, b)

    np.testing.assert_allclose(other_U @ x, b)

    # Test that solver is jittable and differentiable.
    jax.jit(solver)(other_U_T.data, b)
    jax.jacobian(solver)(other_U_T.data, b)

    # Test that cholesky_factor is differentiable by computing its Jacobian.
    jax.jacfwd(solver)(other_U_T.data, b)

  def test_solver_asserts(self):
    rng = np.random.default_rng(0)
    L = random_sparse_L(7, num_entries=10, rng=rng)
    U = scipy.sparse.csc_matrix(L.T)

    with self.assertRaisesRegex(AssertionError, 'populated diagonal entries'):
      L_singular = L.copy()
      L_singular[2, 2] = 0.0
      L_singular.eliminate_zeros()
      sparse.tril_solver(L_singular)

    with self.assertRaisesRegex(AssertionError, 'populated diagonal entries'):
      U_singular = U.copy()
      U_singular[2, 2] = 0.0
      U_singular.eliminate_zeros()
      sparse.triu_solver(U_singular)

    with self.assertRaisesRegex(AssertionError, 'Only square'):
      sparse.tril_solver(L[:-1, :])

    with self.assertRaisesRegex(AssertionError, 'Only square'):
      sparse.tril_solver(L[:, :-1])

    with self.assertRaisesRegex(AssertionError, 'lower triangular'):
      sparse.tril_solver(U)

    with self.assertRaisesRegex(AssertionError, 'upper triangular'):
      sparse.triu_solver(L)

  def test_Q_add(self):
    rng = np.random.default_rng(0)
    L = np.abs(random_sparse_L(7, num_entries=10, rng=rng))
    Q = L @ L.T  # PSD + nonzero diagonal entries
    # Somehow, Q from above does not have sorted indices.
    Q = scipy.sparse.csc_matrix.sorted_indices(Q)
    Q_data = jnp.array(Q.data)
    b = rng.normal(size=Q.shape[0])

    Q_add = sparse.add_fn(Q)
    # Test that adding a diagonal matrix to Q works.
    Q_add_data = Q_add(Q_data, b)
    np.testing.assert_allclose(Q_add_data, (Q + scipy.sparse.diags(b)).data)

  def test_Q_multiply(self):
    rng = np.random.default_rng(0)
    L = np.abs(random_sparse_L(7, num_entries=10, rng=rng))
    Q = L @ L.T  # PSD + nonzero diagonal entries
    b = rng.normal(size=Q.shape[0])

    Q_multiply = sparse.multiply_fn(Q)
    # Test that right multiplication of Q by a dense vector works.
    Q_T_data = jnp.array(scipy.sparse.csc_matrix(Q.T).data)
    y = Q_multiply(Q_T_data, b)
    np.testing.assert_allclose(Q @ b, y)

  def test_logdet_fn(self):
    rng = np.random.default_rng(0)
    L = np.abs(random_sparse_L(7, num_entries=10, rng=rng))  # Diag entries > 0
    L_log_det = sparse.logdet_fn(L)
    L_data = jnp.array(L.data)
    det = L_log_det(L_data)

    # Test that computing the log determinant of L works.
    np.testing.assert_allclose(np.log(scipy.linalg.det(L.todense())), det)

    with self.assertRaisesRegex(AssertionError, 'L must be triangular'):
      sparse.logdet_fn(L @ L.T)


class SPDSparsityStructureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A running example for this test case.
    self.n = 40
    edges = [(i, i + 1) for i in range(self.n - 1)]
    self.sparsity_structure = sparse.SPDSparsityStructure(edges)

    entries = np.arange(self.n)
    entry_locations = [(i, i + 1) for i in range(self.n // 2)] + [
        (i, i - 1) for i in range(self.n // 2, self.n)
    ]
    rows, cols = zip(*entry_locations)
    self.mat = scipy.sparse.coo_matrix(
        (entries, (rows, cols)), shape=(self.n, self.n)
    )
    self.data = self.sparsity_structure.data_from_scipy(self.mat)

  def test_data_size(self):
    # A tri-diagonal n x n matrix has 3n - 2 entries.
    self.assertEqual(3 * self.n - 2, self.sparsity_structure.data_size)
    self.assertLen(self.data, self.sparsity_structure.data_size)

  def test_to_from_data_vector(self):
    np.testing.assert_equal(
        self.mat.todense(),
        self.sparsity_structure.data_to_scipy(self.data).todense(),
    )

  def test_transpose(self):
    data_T = self.sparsity_structure.transpose(self.data)
    np.testing.assert_equal(
        self.sparsity_structure.data_to_scipy(self.data).T.todense(),
        self.sparsity_structure.data_to_scipy(data_T).todense(),
    )

  def test_to_data_vector_wrong_sparsity_structure(self):
    mat = scipy.sparse.coo_matrix(
        ([1.0], ([0], [self.n - 1])), shape=(self.n, self.n)
    )
    with self.assertRaisesRegex(ValueError, 'edge is not in the graph'):
      _ = self.sparsity_structure.data_from_scipy(mat)

  def test_to_data_vector_wrong_size(self):
    with self.assertRaisesRegex(AssertionError, 'Wrong data size'):
      _ = self.sparsity_structure.data_to_scipy(self.data[:-1])

  def test_to_spd_matrix_wrong_size(self):
    with self.assertRaisesRegex(AssertionError, 'Wrong data size'):
      _ = self.sparsity_structure.to_spd_matrix(self.data[:-1])

  def test_to_spd_matrix_not_symmetric(self):
    with self.assertRaisesRegex(AssertionError, 'not.* a symmetric matrix'):
      _ = self.sparsity_structure.to_spd_matrix(self.data)

  def test_to_spd_matrix_not_positive(self):
    data = (self.data + self.sparsity_structure.transpose(self.data)) / 2
    with self.assertRaisesRegex(AssertionError, 'not.* positive definite'):
      _ = self.sparsity_structure.to_spd_matrix(data)

  def test_to_spd_matrix(self):
    # Make positive definite by beefing up the diagonal.
    data = (self.data + self.sparsity_structure.transpose(self.data)) / 2
    data = data + self.n * self.sparsity_structure.data_from_scipy(
        scipy.sparse.eye(self.n)
    )
    _ = self.sparsity_structure.to_spd_matrix(data)


class SPDMatrixTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A running example for this test case.
    self.rng = np.random.default_rng(0)
    mat = random_sparse_L(n=12, num_entries=20, rng=self.rng)
    self.scipy_Q = mat @ mat.T
    self.Q = sparse.SPDMatrix.from_scipy(self.scipy_Q)

  def test_to_from_scipy(self):
    np.testing.assert_equal(self.scipy_Q.todense(), self.Q.to_scipy().todense())

  def test_logdet(self):
    np.testing.assert_almost_equal(
        np.log(np.linalg.det(self.scipy_Q.todense())), np.array(self.Q.logdet)
    )

  def test_matmul(self):
    vec = self.rng.normal(size=self.Q.shape[1])
    np.testing.assert_almost_equal(self.scipy_Q @ vec, self.Q @ vec)
    _ = jax.jit(self.Q.__matmul__)(vec)
    _ = jax.jacobian(self.Q.__matmul__)(vec)

  def test_add_diag(self):
    vec = self.rng.normal(size=self.Q.shape[1])
    np.testing.assert_almost_equal(
        self.scipy_Q + np.diag(vec), self.Q.add_diag(vec).to_scipy().todense()
    )
    _ = jax.jit(self.Q.add_diag)(vec)
    _ = jax.jacobian(self.Q.add_diag)(vec)

  def test_solve_LT(self):
    vec = self.rng.normal(size=self.Q.shape[1])
    x = self.Q.solve_LT(vec)
    np.testing.assert_almost_equal(x @ (self.Q @ x), vec @ vec)
    _ = jax.jit(self.Q.solve_LT)(vec)
    _ = jax.jacobian(self.Q.solve_LT)(vec)

  def test_solve(self):
    vec = self.rng.normal(size=self.Q.shape[1])
    x = self.Q.solve(vec)
    np.testing.assert_almost_equal(self.Q @ x, vec)
    _ = jax.jit(self.Q.solve)(vec)
    _ = jax.jacobian(self.Q.solve)(vec)


if __name__ == '__main__':
  absltest.main()
