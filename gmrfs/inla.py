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

"""INLA marginal posterior calculation."""

from typing import Any, Callable
from gmrfs import gmrf
import jax
import jax.numpy as jnp
import jaxopt


def vmap_all(func):
  """Vectorized scaler function; assumes all arguments are the same shape."""

  def vfunc(*args):
    shape = jnp.array(args[0]).shape
    args = [jnp.array(arg).ravel() for arg in args]
    result = jax.vmap(func)(*args)
    return result.reshape(shape)

  return vfunc


class Inla:
  """INLA marginal posterior calculation.

  This class estimates an (unnormalized) log marginal posterior function
  log(P(parameters | measurement))
  given the following inputs:
  * a prior on parameters, log_prior_fn(params),
  * a GMRF model for a latent state, i.e. parameterized precision Q(params) and
    mu(params), with a latent state x assumed to be drawn from the
    corresponding multivariate gaussian, and
  * a measurement model expressing the probability of seeing the measurement y
    assuming a latent state x, measurement_log_prob_fn(x, y). This is assumed to
    be the same function at each node of the graph (NOTE: this assumption could
    be relaxed if needed).

  The basic strategy is to first use Bayes' law:
    P(params | y) = P(y | params) * P(params) / P(y)
  then use "Bayes' second law" to compute P(y | params):
    P(y | params) = P(y | x, params) * P(x | params) / P(x | y, params)
  The factors in the numerator are straightforward, and the denominator
  P(x | y, params) is estimated by Laplace approximation. We are free to choose
  any x we like for the RHS, but the Laplace approximation is assumed to be best
  if x is chosen to maximize P(x | y, params). The quadratic approximation
  needed for the Laplace approximation is the same one needed to run Newton's
  algorithm to find this x. Let f(x) denote a vectorized
  measurement_log_prob_fn(x, y), and fp(x) and fpp(x) denote its vectorized
  first and second derivatives with respect to x ("vectorized" here is just
  building in the assumption that each coordinate of y depends only on the
  corresponding coordinate of x). The second order approximation at a given x0
  is
  log(P(x | y, params)) = log(P(y | x, params)) +
                          log(P(x | params)) +
                          const
    = f(x0) + (x - x0) * fp(x0) + 0.5 (x - x0)**2 * fpp(x0) +  # P(y| x, params)
      -0.5 * (x - mu).T @ Q @ (x - mu) + 0.5 * logdet(Q) +     # P(x | params)
      const
    = -0.5 * x.T @ (-fpp(x0) + Q) @ x +              # group quadratic terms
      x.T @ (Q @ mu + fp(x0) - x0 * fpp(x0)) +       # group linear terms
      const
  """

  def __init__(
      self,
      gmrf_model: gmrf.Model,
      measurement_log_prob_fn: Callable[[Any], jnp.ndarray],
      log_prior_fn: Callable[[Any], jnp.ndarray] = lambda params: 0.0,
  ):
    """.

    Args:
      gmrf_model: a gmrf.Model. This corresponds to P(x| params).
      measurement_log_prob_fn: callable which takes a latent value and
        measurement and returns a float. This corresponds to P(y | x).
      log_prior_fn: callable which takes an argument of the same shape as
        gmrf_model.init_params() and returns a float/jnp.ndarray. This
        corresponds to P(params). Uses a uniform prior by default.
    """
    self.log_prior_fn = log_prior_fn
    self.gmrf_model = gmrf_model
    self.f = vmap_all(measurement_log_prob_fn)
    self.fp = vmap_all(jax.grad(measurement_log_prob_fn))
    self.fpp = vmap_all(jax.grad(jax.grad(measurement_log_prob_fn)))

  def log_marginal_likelihood(self, params, y: jnp.ndarray) -> jnp.ndarray:
    """Approximate log(P(y|params)), up to a constant independent of params."""
    gaussian = self.gmrf_model(params)

    # Choose x to maximize P(x|y, params) by iterative gaussian approximation.
    # NOTE: if we find that we're getting bad behavior, we might need to
    #   regularize these newton steps, either limiting step size or adding a
    #   multiple of the identity to the covariance.

    def newton_step(x, y, gaussian):
      return self._conditional_gaussian_approximation(x, y, gaussian).mean

    fpi = jaxopt.FixedPointIteration(newton_step)
    x = fpi.run(gaussian.mean, y, gaussian).params

    log_laplace_approx = self._conditional_gaussian_approximation(
        x, y, gaussian
    ).logpdf(x)
    # pyformat: disable
    likelihood = (                 # P(y | params) =
        gaussian.logpdf(x)         # P(x | params)
        + self.f(x, y).sum()       # * P(y | x, params)
        - log_laplace_approx       # / P(x | y, params)
    )
    # pyformat: enable
    return likelihood

  def _conditional_gaussian_approximation(
      self, x, y, unconditional_gaussian
  ) -> gmrf.Gaussian:
    """The gaussian approximation of P(x | y, params) at the given x."""
    fpp = self.fpp(x, y)
    precision = unconditional_gaussian.precision.add_diag(-fpp)
    linear_part = (
        unconditional_gaussian.information_vector + self.fp(x, y) - x * fpp
    )
    return gmrf.Gaussian(precision, precision.solve(linear_part))

  def log_marginal_posterior(self, params, y: jnp.ndarray) -> jnp.ndarray:
    """Approximate log(P(params|y)), up to a constant independent of params."""
    return self.log_prior_fn(params) + self.log_marginal_likelihood(params, y)
