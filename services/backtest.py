import numpy as np
import pandas as pd
import itertools
from services.strategies import *
import math

from scipy.stats import gmean

def compute_metrics_official(portfValue, riskFree, turnover):
    portfRets = portfValue.pct_change(1).iloc[1:, :]

    rf = riskFree[
        (riskFree.index >= portfRets.index[0]) &
        (riskFree.index <= portfRets.index[-1])
    ]

    portfExRets = portfRets.subtract(rf, axis=0)

    SR = ((portfExRets + 1).apply(gmean, axis=0) - 1) / portfExRets.std()
    avgTurnover = np.mean(turnover[1:])

    return float(SR.iloc[0]), float(avgTurnover)


def run_backtest(prices, factorReturns, strategy_fn,
                 initialVal=100000, init_window=60, rebal_freq=6):

    prices = prices.copy()
    factorReturns = factorReturns.copy()

    riskFree = factorReturns["RF"]
    factors = factorReturns.drop(columns=["RF"])

    # Asset excess returns for strategy estimation
    assetReturns = prices.pct_change(1).iloc[1:, :]
    assetExcessReturns = assetReturns.subtract(riskFree, axis=0)

    # Align prices with returns
    prices = prices.iloc[1:, :]

    T, n = assetExcessReturns.shape

    NoPeriods = math.ceil((T - init_window) / rebal_freq)

    x = np.zeros([n, NoPeriods])
    x0 = np.zeros([n, NoPeriods])
    currentVal = np.zeros([NoPeriods, 1])
    turnover = np.zeros([NoPeriods, 1])

    portfValue = []

    for t in range(NoPeriods):

        start_idx = init_window + t * rebal_freq
        end_idx = min(start_idx + rebal_freq, T)

        if start_idx >= T:
            break

        calEnd = assetExcessReturns.index[start_idx - 1]

        periodReturns = assetExcessReturns.iloc[:start_idx, :]
        periodFactRet = factors.iloc[:start_idx, :]

        currentPrices = prices.loc[[calEnd]]
        periodPrices = prices.iloc[start_idx:end_idx, :]

        if t == 0:
            currentVal[t] = initialVal
        else:
            currentVal[t] = currentPrices @ NoShares.values.T
            x0[:, t] = currentPrices.values * NoShares.values / currentVal[t]
        
        try:
            x[:, t] = strategy_fn(periodReturns, periodFactRet, x0)
        except TypeError:
            x[:, t] = strategy_fn(periodReturns, periodFactRet)

       

        if t > 0:
            turnover[t] = np.sum(np.abs(x[:, t] - x0[:, t]))

        NoShares = x[:, t] * currentVal[t] / currentPrices

        portfValue.append(periodPrices @ NoShares.values.T)

    portfValue = pd.concat(portfValue, axis=0)

    sharpe, avg_turnover = compute_metrics_official(
        portfValue,
        riskFree,
        turnover
    )

    summary = {
        "sharpe": sharpe,
        "avg_turnover": avg_turnover
    }

    return portfValue, summary

# def run_backtest(prices, factorReturns, strategy_fn,
#                  init_window=60, rebal_freq=6):
#     """
#     Walk-forward backtest engine. Works with any strategy.

#     :param prices:         T x n DataFrame of adjusted closing prices
#     :param factorReturns:  T x p DataFrame of factor returns, aligned to prices
#     :param strategy_fn:    callable(periodReturns, periodFactRet) -> weights (n,)
#     :param init_window:    months reserved for initial calibration
#     :param rebal_freq:     months between rebalancing
#     :return:               results (list of dicts), summary (dict)
#     """
#     returns = prices.pct_change().dropna()
#     T, n = returns.shape

#     results = []
#     current_weights = None

#     for t in range(init_window, T, rebal_freq):
#         period_ret  = returns.iloc[:t, :]
#         period_fact = factorReturns.iloc[:t, :]

#         new_weights = strategy_fn(period_ret, period_fact)

#         turnover = None
#         if current_weights is not None:
#             turnover = np.sum(np.abs(new_weights - current_weights))

#         current_weights = new_weights

#         end = min(t + rebal_freq, T)
#         for s in range(t, end):
#             r = returns.iloc[s, :].values
#             results.append({
#                 'date':     returns.index[s],
#                 'return':   current_weights @ r,
#                 'turnover': turnover if s == t else None
#             })

#     sharpe, avg_turnover = compute_metrics(results)
#     summary = {'sharpe': sharpe, 'avg_turnover': avg_turnover}

#     print(f"Sharpe ratio:  {sharpe:.4f}")
#     print(f"Avg turnover:  {avg_turnover:.4f}")

#     return results, summary




def make_ridge_lw(params):
    prev_w = [None]

    def strategy_fn(periodReturns, periodFactRet):
        strat = RidgeLedoitWolf_MVO(
            NumObs=params.get("NumObs", 48),
            ridge_alpha=params.get("ridge_alpha", 0.1),
            lw_shrink_weight=params.get("lw_shrink_weight", 0.7),
            turnover_penalty=params.get("turnover_penalty", 0.1),
            prev_weights=prev_w[0]
        )

        x = strat.execute_strategy(periodReturns, periodFactRet)

        prev_w[0] = x

        return x

    return strategy_fn


def make_ols_mvo(params):
    def strategy_fn(periodReturns, periodFactRet):
        strat = OLS_MVO(
            NumObs=params.get("NumObs", 36),
            risk_aversion=params.get("risk_aversion", 5.0),
            max_weight=params.get("max_weight", 0.25)
        )
        return strat.execute_strategy(periodReturns, periodFactRet)

    return strategy_fn


def make_historical_mvo(params):
    def strategy_fn(periodReturns, periodFactRet):
        strat = HistoricalMeanVarianceOptimization(
            NumObs=params.get("NumObs", 36),
            risk_aversion=params.get("risk_aversion", 5.0),
            max_weight=params.get("max_weight", 0.25)
        )
        return strat.execute_strategy(periodReturns, periodFactRet)

    return strategy_fn


def make_equal_weight(params):
    """Factory for equal weight — no parameters needed."""
    def strategy_fn(periodReturns, periodFactRet):
        return equal_weight(periodReturns)

    return strategy_fn

def make_risk_parity(params):
    def strategy_fn(periodReturns, periodFactRet):
        strat = RiskParity(
            NumObs=params.get("NumObs", 36),
            covariance_method=params.get("covariance_method", "ledoit_wolf")
        )
        return strat.execute_strategy(periodReturns, periodFactRet)
    return strategy_fn


def make_historical_max_sharpe(params):
    def strategy_fn(periodReturns, periodFactRet):
        strat = HistoricalMaxSharpe(
            NumObs=params.get("NumObs", 36),
            max_weight=params.get("max_weight", 0.25)
        )
        return strat.execute_strategy(periodReturns, periodFactRet)
    return strategy_fn


def make_project_function(params):
    from services.project_function import project_function

    def strategy_fn(periodReturns, periodFactRet, x0=None):
        return project_function(periodReturns, periodFactRet, x0)

    return strategy_fn



STRATEGY_REGISTRY = {
    "ridge_lw": make_ridge_lw,
    "ols_mvo": make_ols_mvo,
    "historical_mvo": make_historical_mvo,
    "equal_weight": make_equal_weight,
    "risk_parity": make_risk_parity,
    "historical_max_sharpe": make_historical_max_sharpe,
    "project_function": make_project_function,
}


def grid_search(prices, factorReturns, strategy_name, param_grid,
                init_window=60, rebal_freq=6,
                train_end=None, val_end=None,
                turnover_weight=0.05):
    """
    Generic grid search over any registered strategy's hyperparameters.

    :param prices:          T x n DataFrame of adjusted closing prices
    :param factorReturns:   T x p DataFrame of factor returns
    :param strategy_name:   string key from STRATEGY_REGISTRY
    :param param_grid:      dict of param_name -> list of values to search
    :param init_window:     months for initial calibration
    :param rebal_freq:      months between rebalancing
    :param train_end:       row index for end of training period (default 60% of T)
    :param val_end:         row index for end of validation period (default 80% of T)
    :return:                best_params (dict), test_summary (dict)
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. "
                         f"Available: {list(STRATEGY_REGISTRY.keys())}")

    factory = STRATEGY_REGISTRY[strategy_name]

    returns = prices.pct_change().dropna()
    T = len(returns)

    train_end = train_end or int(T * 0.6)
    val_end   = val_end   or int(T * 0.8)

    train_prices  = prices.iloc[:train_end + 1]
    val_prices    = prices.iloc[train_end + 1:val_end + 1]
    test_prices   = prices.iloc[val_end + 1:]

    train_factors = factorReturns.iloc[:train_end]
    val_factors   = factorReturns.iloc[train_end:val_end]
    test_factors  = factorReturns.iloc[val_end:]

    keys         = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    print(f"Strategy: {strategy_name}")
    print(f"Running grid search over {len(combinations)} combinations...\n")

    best_score = -np.inf
    best_params = None
    best_summary = None

    for combo in combinations:

        params = dict(zip(keys, combo))

        combined_p = pd.concat([train_prices, val_prices])
        combined_f = pd.concat([train_factors, val_factors])

        _, summary = run_backtest(
            combined_p,
            combined_f,
            strategy_fn=factory(params),
            init_window=train_end,
            rebal_freq=rebal_freq
        )

        score = (
            summary["sharpe"]
            - turnover_weight * summary["avg_turnover"]
        )

        print(
            f"Params: {params}"
            f" -> Sharpe: {summary['sharpe']:.4f}"
            f", Turnover: {summary['avg_turnover']:.4f}"
            f", Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_params = params
            best_summary = summary

    print(f"\nBest params: {best_params}")
    print(f"Best validation Sharpe: {best_summary['sharpe']:.4f}")
    print(f"Best validation Turnover: {best_summary['avg_turnover']:.4f}")
    print(f"Best validation Score: {best_score:.4f}\n")

    # Final test evaluation with best params
    print("Evaluating on test set...")
    combined_p = pd.concat([train_prices, val_prices, test_prices])
    combined_f = pd.concat([train_factors, val_factors, test_factors])

    _, test_summary = run_backtest(
        combined_p, combined_f,
        strategy_fn=factory(best_params),
        init_window=val_end,
        rebal_freq=rebal_freq
    )

    print(f"Test Sharpe:   {test_summary['sharpe']:.4f}")
    print(f"Test Turnover: {test_summary['avg_turnover']:.4f}")

    return best_params, test_summary


def compare_strategies(prices, factorReturns, strategies,
                       init_window=60, rebal_freq=6):
    """
    Runs multiple strategies with fixed params and prints a comparison table.
    Useful for a final side-by-side evaluation before choosing your submission strategy.

    :param prices:         T x n DataFrame of adjusted closing prices
    :param factorReturns:  T x p DataFrame of factor returns
    :param strategies:     dict of {label: (strategy_name, params)}
                           e.g. {'Ridge+LW': ('ridge_lw', {'NumObs': 60, ...}),
                                 'OLS MVO':  ('ols_mvo',  {'NumObs': 36}),
                                 'Equal W':  ('equal_weight', {})}
    :param init_window:    months reserved for calibration
    :param rebal_freq:     months between rebalancing
    :return:               DataFrame of results indexed by strategy label
    """
    rows = []

    for label, (strategy_name, params) in strategies.items():
        print(f"\n--- {label} ---")
        factory = STRATEGY_REGISTRY[strategy_name]

        _, summary = run_backtest(
            prices, factorReturns,
            strategy_fn=factory(params),
            init_window=init_window,
            rebal_freq=rebal_freq
        )

        rows.append({
            'Strategy':     label,
            'Sharpe':       round(summary['sharpe'], 4),
            'Avg Turnover': round(summary['avg_turnover'], 4),
        })

    results_df = pd.DataFrame(rows).set_index('Strategy')
    print("\n=== Strategy Comparison ===")
    print(results_df.to_string())

    return results_df
