from scipy.optimize import minimize


class RiskParity:
    """
    Long-only risk parity strategy.

    Finds portfolio weights such that each asset contributes approximately
    equally to total portfolio variance.
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        :param periodReturns: asset return DataFrame
        :param factorReturns: unused
        :return: x, long-only fully invested portfolio weights
        """

        returns = periodReturns.iloc[-self.NumObs:, :]
        Q = returns.cov().values

        n = Q.shape[0]

        # small ridge term for numerical stability
        Q = Q + 1e-6 * np.eye(n)

        def portfolio_variance(w):
            return w.T @ Q @ w

        def risk_contributions(w):
            port_var = portfolio_variance(w)
            marginal_risk = Q @ w
            rc = w * marginal_risk / port_var
            return rc

        def objective(w):
            rc = risk_contributions(w)
            target = np.ones(n) / n
            return np.sum((rc - target) ** 2)

        constraints = ({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        })

        bounds = [(0, 1) for _ in range(n)]

        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if not result.success:
            # fallback to equal weight if optimization fails
            x = w0
        else:
            x = result.x

        return x