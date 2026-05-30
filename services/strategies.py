import numpy as np
from services.estimators import *
from services.optimization import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x

    
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