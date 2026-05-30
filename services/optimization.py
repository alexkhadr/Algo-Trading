import cvxpy as cp
import numpy as np


def clean_weights(x, n):
    """
    Clean portfolio weights after optimization.

    This function handles numerical issues:
    - tiny negative values
    - weights not summing exactly to 1
    - failed optimization output
    """

    if x is None:
        return np.ones(n) / n

    x = np.asarray(x).flatten()

    # Remove tiny negative values caused by numerical precision
    x[x < 1e-8] = 0

    # If optimization failed or returned invalid values, use equal weights
    if np.sum(x) <= 0 or np.any(np.isnan(x)):
        return np.ones(n) / n

    # Normalize weights to sum to 1
    x = x / np.sum(x)

    return x


def mean_variance_optimization(mu, Q, risk_aversion=5.0, max_weight=0.25):
    """
    Long-only Mean-Variance Optimization.

    Model:

        maximize        mu' x - (risk_aversion / 2) x' Q x

        subject to      sum(x) = 1
                        0 <= x_i <= max_weight

    Inputs:
        mu: expected return vector
        Q: covariance matrix
        risk_aversion: controls the trade-off between return and risk
        max_weight: maximum allocation allowed in one asset

    Output:
        x: portfolio weights
    """

    n = Q.shape[0]
    mu = np.asarray(mu).flatten()

    x = cp.Variable(n)

    objective = cp.Maximize(
        mu @ x - (risk_aversion / 2) * cp.quad_form(x, Q)
    )

    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= max_weight
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(verbose=False)

        return clean_weights(x.value, n)

    except Exception:
        # If the solver fails, use equal weights as a safe fallback
        return np.ones(n) / n


def MVO(mu, Q):
    """
    Main MVO function used by the strategy.

    Recommended setup:
        mu, Q = estimate_mean_and_covariance(returns, method="ledoit_wolf")
        x = MVO(mu, Q)
    """

    x = mean_variance_optimization(
        mu=mu,
        Q=Q,
        risk_aversion=5.0,
        max_weight=0.25
    )

    return x

