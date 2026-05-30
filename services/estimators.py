import numpy as np


def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q


def ridge_factor_model(returns, factRet, alpha=0.1):
    """
    Ridge regression factor model — numpy only implementation.
    Solves: B = (X'X + alpha * I)^{-1} X'Y
    The regularization shrinks betas toward zero, reducing sensitivity
    to multicollinearity between factors.

    :param returns:  T x n DataFrame of asset returns
    :param factRet:  T x p DataFrame of factor returns
    :param alpha:    L2 regularization strength (larger = more shrinkage)
    :return:         mu (n x 1), Q (n x n), V (p x n loadings), F (p x p), D (n x n)
    """
    T, p = factRet.shape

    # Center factors and returns (Ridge without intercept on demeaned data
    # is equivalent to Ridge with intercept — avoids regularizing the intercept)
    f_mean = factRet.mean(axis=0).values        # (p,)
    r_mean = returns.mean(axis=0).values        # (n,)

    F_centered = factRet.values - f_mean        # (T, p)
    R_centered = returns.values - r_mean        # (T, n)

    # Ridge normal equations: (F'F + alpha * I) V = F'R
    A = F_centered.T @ F_centered + alpha * np.eye(p)
    V = np.linalg.solve(A, F_centered.T @ R_centered)   # (p, n)

    # Recover intercepts: a = r_mean - f_mean @ V
    a = r_mean - f_mean @ V                             # (n,)

    # Residual variance (idiosyncratic)
    R_hat = F_centered @ V + r_mean            # (T, n)
    ep = returns.values - R_hat
    sigma_ep = 1 / (T - p - 1) * np.sum(ep ** 2, axis=0)
    D = np.diag(sigma_ep)

    # Factor covariance
    F_cov = factRet.cov().values               # (p, p)

    # Asset expected returns and covariance
    f_bar = f_mean.reshape(-1, 1)
    mu = a.reshape(-1, 1) + V.T @ f_bar
    Q = V.T @ F_cov @ V + D
    Q = (Q + Q.T) / 2

    return mu, Q, V, F_cov, D


def ledoit_wolf_covariance(returns, shrink_target=None, shrink_weight=0.5):
    """
    Ledoit-Wolf covariance estimator — numpy only implementation.

    Two modes:
      1. shrink_target=None:
         Analytic LW formula (Ledoit & Wolf 2004) — shrinks sample covariance
         toward a scaled identity matrix with optimal alpha computed analytically.

      2. shrink_target provided (e.g. factor model Q):
         Manual blend: Q = (1 - w) * S + w * target
         This lets you use the factor model covariance as the shrinkage target,
         which is more economically motivated than shrinking toward identity.

    :param returns:        T x n array or DataFrame
    :param shrink_target:  (n x n) structured target matrix, or None
    :param shrink_weight:  blending weight for custom target mode (0 to 1)
    :return:               Q (n x n) shrunk covariance matrix
    """
    if hasattr(returns, 'values'):
        R = returns.values
    else:
        R = np.array(returns)

    T, n = R.shape
    S = np.cov(R.T)     # sample covariance (n x n)

    if shrink_target is not None:
        # Custom target mode: simple convex blend
        Q = (1 - shrink_weight) * S + shrink_weight * shrink_target

    else:
        # Analytic Ledoit-Wolf (Oracle Approximating Shrinkage)
        # Shrinks toward mu_hat * I where mu_hat = trace(S) / n

        # Shrinkage target: scaled identity
        mu_hat = np.trace(S) / n
        target = mu_hat * np.eye(n)

        # Compute optimal shrinkage intensity analytically
        # Following Ledoit & Wolf (2004), simplified form:
        X = R - R.mean(axis=0)   # demeaned (T x n)

        # delta^2: squared Frobenius norm of S - target
        delta2 = np.linalg.norm(S - target, 'fro') ** 2

        # beta^2: asymptotic variance term
        # Approximated as: (1/T^2) * sum_t ||x_t x_t' - S||^2_F
        beta2 = 0.0
        for t in range(T):
            x = X[t:t+1, :].T          # (n, 1)
            outer = x @ x.T            # (n, n)
            beta2 += np.linalg.norm(outer - S, 'fro') ** 2
        beta2 /= T ** 2

        # Optimal shrinkage intensity (clamped to [0, 1])
        alpha_lw = min(beta2 / delta2, 1.0) if delta2 > 0 else 0.0

        Q = (1 - alpha_lw) * S + alpha_lw * target

    Q = (Q + Q.T) / 2   # enforce symmetry
    return Q