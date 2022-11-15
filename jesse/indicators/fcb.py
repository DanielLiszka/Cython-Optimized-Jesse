from stock_indicators import indicators

import numpy as np

from jesse.helpers import slice_candles,get_candle_source


def fcb(candles: np.ndarray, window_span:int=2,source_type: str = "close",sequential: bool = False):
    """
    fractal chaos bands
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_fcb(source,window_span)

    return res if sequential else res[-1]