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

    def __init__(self, NumObs=36, risk_aversion=5.0, max_weight=0.25):
        self.NumObs = NumObs
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    def execute_strategy(self, periodReturns, factorReturns=None):

        returns = periodReturns.iloc[-self.NumObs:, :]

        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values

        x = MVO(
            mu,
            Q,
            risk_aversion=self.risk_aversion,
            max_weight=self.max_weight
        )

        return x


class OLS_MVO:

    def __init__(self, NumObs=36, risk_aversion=5.0, max_weight=0.25):
        self.NumObs = NumObs
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    def execute_strategy(self, periodReturns, factorReturns):

        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]

        mu, Q = OLS(returns, factRet)

        x = MVO(
            mu,
            Q,
            risk_aversion=self.risk_aversion,
            max_weight=self.max_weight
        )

        return x

    
class RiskParity:
    """
    Long-only risk parity strategy.

    Uses either:
    - sample covariance
    - Ledoit-Wolf covariance
    """

    def __init__(self, NumObs=36, covariance_method="ledoit_wolf"):
        self.NumObs = NumObs
        self.covariance_method = covariance_method

    def execute_strategy(self, periodReturns, factorReturns=None):

        returns = periodReturns.iloc[-self.NumObs:, :]

        if self.covariance_method == "sample":
            Q = returns.cov().values

        elif self.covariance_method == "ledoit_wolf":
            Q = ledoit_wolf_covariance(returns)

        else:
            raise ValueError("covariance_method must be 'sample' or 'ledoit_wolf'")

        n = Q.shape[0]

        Q = (Q + Q.T) / 2
        Q = Q + 1e-6 * np.eye(n)

        def portfolio_variance(w):
            return w.T @ Q @ w

        def risk_contributions(w):
            port_var = portfolio_variance(w)
            marginal_risk = Q @ w
            return w * marginal_risk / port_var

        def objective(w):
            rc = risk_contributions(w)
            target = np.ones(n) / n
            return np.sum((rc - target) ** 2)

        constraints = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }

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

        if result.success:
            return result.x
        else:
            return w0
    

class RidgeLedoitWolf_MVO:
    """
    Strategy combining:
      - Ridge regression on Fama-French factors for expected returns (mu)
      - Ledoit-Wolf covariance shrunk toward the factor model covariance (Q)
      - Max-Sharpe MVO with turnover penalty
    """

    def __init__(self, NumObs=48, ridge_alpha=0.1, lw_shrink_weight=0.7,
                 turnover_penalty=0.1, prev_weights=None):
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

        # Step 1: Ridge factor model → mu and factor-model covariance as LW target
        mu, Q_lw_target, _, _, _ = ridge_factor_model(returns, factRet,
                                                       alpha=self.ridge_alpha)

        # Step 2: Apply Ledoit-Wolf shrinkage toward the factor model covariance
        Q_lw = ledoit_wolf_covariance(returns,
                                      shrink_target=Q_lw_target,
                                      shrink_weight=self.lw_shrink_weight)

        # Step 3: MVO — just pass Q_lw directly, no extra parameters needed
        x = MVO(mu, Q_lw,
                prev_weights=self.prev_weights,
                turnover_penalty=self.turnover_penalty)

        return x
    
    
    
class HistoricalMaxSharpe:
    """
    Long-only maximum Sharpe strategy using historical mean returns
    and Ledoit-Wolf covariance.
    """

    def __init__(self, NumObs=36, max_weight=0.25):
        self.NumObs = NumObs
        self.max_weight = max_weight

    def execute_strategy(self, periodReturns, factorReturns=None):

        returns = periodReturns.iloc[-self.NumObs:, :]

        mu = returns.mean(axis=0).values
        Q = ledoit_wolf_covariance(returns)

        x = max_sharpe_optimization(
            mu=mu,
            Q=Q,
            rf=0.0,
            max_weight=self.max_weight
        )

        return x