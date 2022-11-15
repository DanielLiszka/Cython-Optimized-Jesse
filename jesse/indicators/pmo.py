from stock_indicators import indicators

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles,get_candle_source


def pmo(candles: np.ndarray, length1:int=35,length2:int=20,length3:int=10,source_type: str = "close",sequential: bool = False):
    """
    price momentum oscillator
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_pmo(source,length1,length2,length3)

    return res if sequential else res[-1]