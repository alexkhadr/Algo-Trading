import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


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


def mean_variance_optimization(mu, Q, risk_aversion=5.0, max_weight=0.25,
                                prev_weights=None, turnover_penalty=0.0):
    """
    Long-only Mean-Variance Optimization.

    Model:

        maximize        mu' x - (risk_aversion / 2) * x' Q x - lambda * ||x - x_prev||_1

        subject to      sum(x) = 1
                        0 <= x_i <= max_weight

    :param mu:               (n,) expected return vector
    :param Q:                (n x n) covariance matrix — pass Q_lw for Ledoit-Wolf
    :param risk_aversion:    controls return vs risk trade-off
    :param max_weight:       maximum allocation per asset
    :param prev_weights:     (n,) weights from previous period (None on first call)
    :param turnover_penalty: lambda >= 0, scales L1 turnover cost in objective
    :return:                 x (n,) cleaned portfolio weights
    """

    n = Q.shape[0]
    mu = np.asarray(mu).flatten()

    Q = (Q + Q.T) / 2
    Q = Q + 1e-6 * np.eye(n)

    x = cp.Variable(n)

    objective = mu @ x - (risk_aversion / 2) * cp.quad_form(x, Q)

    if prev_weights is not None and turnover_penalty > 0:
        prev_weights = np.asarray(prev_weights).flatten()
        objective = objective - turnover_penalty * cp.norm1(x - prev_weights)

    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= max_weight
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)

    try:
        problem.solve(verbose=False)
        return clean_weights(x.value, n)

    except Exception:
        return np.ones(n) / n


def MVO(mu, Q, prev_weights=None, turnover_penalty=0.0,
        risk_aversion=5.0, max_weight=0.25):
    """
    Main MVO function used by the strategy. Pass Q_lw (Ledoit-Wolf covariance) directly
    if using LW shrinkage — no extra parameters needed here.

    :param mu:               (n x 1) or (n,) expected returns
    :param Q:                (n x n) covariance matrix
    :param prev_weights:     (n,) previous weights for turnover penalty
    :param turnover_penalty: lambda scaling the L1 turnover cost
    :param risk_aversion:    MVO risk aversion parameter
    :param max_weight:       max weight per asset
    :return:                 x (n,) portfolio weights
    """

    x = mean_variance_optimization(
        mu=mu,
        Q=Q,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        prev_weights=prev_weights,
        turnover_penalty=turnover_penalty
    )

    return x




def max_sharpe_optimization(mu, Q, rf=0.0, max_weight=0.25):
    """
    Long-only maximum Sharpe ratio portfolio.

    maximize (mu'x - rf) / sqrt(x'Qx)

    subject to:
        sum(x) = 1
        0 <= x_i <= max_weight
    """

    n = Q.shape[0]
    mu = np.asarray(mu).flatten()

    Q = (Q + Q.T) / 2
    Q = Q + 1e-6 * np.eye(n)

    def negative_sharpe(x):
        port_return = mu @ x
        port_vol = np.sqrt(x.T @ Q @ x)

        if port_vol <= 0:
            return 1e6

        return -((port_return - rf) / port_vol)

    constraints = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1
    }

    bounds = [(0, max_weight) for _ in range(n)]

    x0 = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )

    if result.success:
        return clean_weights(result.x, n)
    else:
        return np.ones(n) / n