from stock_indicators import indicators

import numpy as np

from jesse.helpers import slice_candles,get_candle_source


def chandelier(candles: np.ndarray, length:int=22, multiplier:float=3,source_type: str = "close",sequential: bool = False):
    """
    chandelier exit
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_chandelier(source,length,multiplier)

    return res if sequential else res[-1]