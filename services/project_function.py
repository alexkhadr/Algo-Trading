from services.strategies import *


def project_function(periodReturns, periodFactRet, x0=None):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = RidgeLedoitWolf_MVO()
    # Strategy = HistoricalMaxSharpe(
    #     NumObs=36,
    #     max_weight=0.25
    # )

    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x
