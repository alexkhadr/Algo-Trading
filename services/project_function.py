from services.strategies import *
import numpy as np


def project_function(periodReturns, periodFactRet, x0=None):
    """
    Main project function.

    This function is called by the main file.
    It receives historical returns, factor returns, and previous/current portfolio weights.
    """

    prev_weights = None

    if x0 is not None:
        x0_array = np.asarray(x0)

        # If x0 is a matrix n x NoPeriods, find the last non-zero column
        if x0_array.ndim == 2:
            col_sums = np.sum(np.abs(x0_array), axis=0)
            valid_cols = np.where(col_sums > 0)[0]

            if len(valid_cols) > 0:
                prev_weights = x0_array[:, valid_cols[-1]]

        # If x0 is already a vector
        elif x0_array.ndim == 1 and np.sum(np.abs(x0_array)) > 0:
            prev_weights = x0_array

    Strategy = RidgeLedoitWolf_MVO(
        NumObs=48,
        ridge_alpha=0.1,
        lw_shrink_weight=0.7,
        turnover_penalty=0.0,
        prev_weights=prev_weights
    )

    x = Strategy.execute_strategy(periodReturns, periodFactRet)

    return x
