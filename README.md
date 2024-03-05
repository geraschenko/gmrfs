# Gaussian Markov Random Fields (GMRFs) and Integrated Nested Laplace Approximation (INLA)

## Installation

To be able to pip install scikit-sparse, you may need to first install
libsuitesparse-dev:

```
sudo apt install libsuitesparse-dev
```

## Overview

The primary goal of this library is the method `Inla.log_marginal_posterior`,
which estimates the parameters of a Gaussian Markov Random Field (GMRF) from
observed data. The user supplies

1.  A parameterized GMRF, i.e. for parameters $\theta$, a mean vector
    $\mu(\theta)$ and a sparse precision matrix $Q(\theta)$. This is specified
    as an instance of `gmrf.Model`. An unobserved state $x$ is assumed to be
    sampled from the multivariate normal with mean $\mu$ and covariance
    $Q^{-1}$.
2.  A prior on parameters, $p(\theta)$.
3.  A measurement vector $y$.
4.  A measurment model: $p(y_i|x_i) = p(y_i|x_i, \theta)$. Our implementation
    assumes this distribution is the same for all $i$, and that it is
    independent of $\theta$, but all that's technically needed is that $y_i$ is
    independent of $x_j$ for $j\neq i$.

Given these inputs, we compute an estimate of the marginal posterior $p(\theta|
y)$, up to a normalization constant independent of $\theta$.

Our implementation is in JAX, so the marginal posterior can be automatically
differentiated with respect to $\theta$, so gradient-based sampling methods
(like HMC) to sample from $p(\theta| y)$.

### How it works

The basic strategy is to use the following formula, which holds for any value of
$x$:

$$
p(\theta|y) = \frac{p(y|x, \theta) p(x|\theta) p(\theta)}{p(x|y, \theta)} \times \frac{1}{p(y)}
$$

The GMRF specifies $p(x|\theta)$, the prior $p(\theta)$ is supplied, the
measurement model gives us $p(y|x,\theta)$, and the normalizing constant $p(y)$
is ignored since $y$ is fixed.

This just leaves $p(x| y, \theta)$, which we estimate with
[Laplace's method](https://en.wikipedia.org/wiki/Laplace%27s_method). Denoting
$f(x) = \log(p(y|x))$, we have the following approximation around any given
$\hat x$:

$$\begin{align}
  \log(p(x|y, \theta))
  &= \log( p(y|x, \theta)) + \log(p(x|\theta)) + \text{const} \\
  &= f(x) -\frac 12 (x-\mu)^T Q(x-\mu) + \frac 12\log\det(Q) + \text{const} \\
  &\approx f(\hat x) + (x-\hat x)f'(\hat x) + \frac 12 (x-\hat x)^2 f''(\hat x) -\frac 12 (x-\mu)^T Q(x-\mu) + \frac 12\log\det(Q) +
\text{const} \\
  &= -\frac 12 x^T(-f''(\hat x) + Q)x + x^T(Q\mu + f'(\hat x) -
\hat xf''(\hat x)) + \frac 12\log\det(Q) + \text{const}
\end{align}$$

The constant is then be determined by using the fact that this must be a
probability distribution in $x$ (i.e. use the value of the constant from the log
probability function of the gaussian with the same quadratic and linear terms).

What value of $\hat x$ should we choose? We think the Laplace approximation
above is best when $\hat x$ is chosen to maximize $p(\hat x|y, \theta)$. We can
find this $\hat x$ with Newton's method, which amounts to iteratively solving
for the maximum in the very same quadratic approximation. That is, we set $\hat
x_0 = \mu$ and iteratively solve for $\hat x_{i+1}$ in

$$
(Q - f''(\hat x_i))\hat x_{i+1} = Q\mu + f'(\hat x_i) - \hat x_i f''(\hat x_i)
$$

until stabilization.

### Corresponding code, and a critical caveat

This library provides the low-level methods required to run the computation
above. `gmrf.Model` provides a class for parameterizing a GMRF (i.e. a
multivariate gaussian with sparse precision matrix). The sparse precision
matrices are implemented with the class `sparse.SPDMatrix`, which supports
adding to the diagonal (for constructing $Q - f''(\hat x)$ ), multilpication by
a vector (for computing $Q\mu$), solving linear systems (for doing a step of
Newton's algorithm), and computing $\log\det Q$.

Linear solves and log determinant calculations are powered by sparse Cholesky
factorization (`sparse.sparse_cholesky_fn`) and sparse triangular solves
(`sparse.tril_solver`, `sparse.triu_solver`). These methods implement the naive
algorithm for Cholesky factorization or triangular solves, taking advantage of
sparsity, but looping over non-zero entries. This library would be substantially
improved if this loop could be implemented with a `jax.lax.scan`. The
fundamental challenge is that `jax.lax.scan` requires each operation in the loop
operate on arrays of the same shape. Since *some* of the rows of the Cholesky
triangle typically have a large number of non-zero entries, the straight-forward
`jax.lax.scan` approach would give up the benefits of sparsity.
