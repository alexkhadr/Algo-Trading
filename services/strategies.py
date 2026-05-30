import numpy as np
from services.estimators import *
from services.optimization import *
from scipy.optimize import minimize



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
    

class RidgeLedoitWolf_MVO:
    """
    Strategy combining:
      - Ridge regression on Fama-French factors for expected returns (mu)
      - Ledoit-Wolf covariance shrunk toward the factor model covariance (Q)
      - Max-Sharpe MVO with turnover penalty
    """

    def __init__(self, NumObs=60, ridge_alpha=0.1, lw_shrink_weight=0.5,
                 turnover_penalty=0.5, prev_weights=None):
        """
        :param NumObs:            rolling window length in months
        :param ridge_alpha:       Ridge L2 regularization strength
        :param lw_shrink_weight:  weight on factor model target in LW blend (0=pure sample, 1=pure factor)
        :param turnover_penalty:  lambda in objective: min variance - mu'w + lambda*||w - w_prev||_1
        :param prev_weights:      previous portfolio weights (updated externally each period)
        """
        self.NumObs = NumObs
        self.ridge_alpha = ridge_alpha
        self.lw_shrink_weight = lw_shrink_weight
        self.turnover_penalty = turnover_penalty
        self.prev_weights = prev_weights  # None on first call → no penalty applied

    def execute_strategy(self, periodReturns, periodFactRet):
        """
        :param periodReturns:   DataFrame, all available returns up to rebalancing date
        :param periodFactRet:   DataFrame, all available factor returns up to rebalancing date
        :return:                x, weight vector (n,)
        """
        # Use the most recent NumObs months
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = periodFactRet.iloc[-self.NumObs:, :]

        # Step 1: Ridge factor model → mu and factor-model covariance as shrinkage target
        mu, Q_factor, _, _, _ = ridge_factor_model(returns, factRet, alpha=self.ridge_alpha)

        # Step 2: Ledoit-Wolf covariance blended toward factor model target
        Q = ledoit_wolf_covariance(returns,
                                   shrink_target=Q_factor,
                                   shrink_weight=self.lw_shrink_weight)

        # Step 3: MVO with optional turnover penalty
        x = MVO(mu, Q,
                prev_weights=self.prev_weights,
                turnover_penalty=self.turnover_penalty)

        return x