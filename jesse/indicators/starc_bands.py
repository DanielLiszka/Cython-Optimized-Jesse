from stock_indicators import indicators

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles,get_candle_source


def starc_bands(candles: np.ndarray, sma_length:int=20, multiplier:float=2.0, atr_period:int=10,source_type: str = "close",sequential: bool = False):
    """
    starc band
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_starc_bands(source,sma_length,multiplier,atr_period)

    return res if sequential else res[-1]