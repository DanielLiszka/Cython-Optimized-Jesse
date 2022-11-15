from stock_indicators import indicators

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles,get_candle_source


def volatility_stop(candles: np.ndarray, length:int=20, multiplier:float=2.5,source_type: str = "close",sequential: bool = False):
    """
    vertical horizontal filter
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_volatility_stop(source,length,multiplier)

    return res if sequential else res[-1]