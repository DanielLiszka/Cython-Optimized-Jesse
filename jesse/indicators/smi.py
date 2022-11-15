from stock_indicators import indicators

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles,get_candle_source


def smi(candles: np.ndarray, length:int=14, smooth_length:int=20, smooth_length_2: int=5, signal_period:int=3,source_type: str = "close",sequential: bool = False):
    """
    stochastic momentum index
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_smi(source,length,smooth_length,smooth_length_2,signal_period)

    return res if sequential else res[-1]