from stock_indicators import indicators
from typing import Union

import numpy as np

from jesse.helpers import slice_candles,get_candle_source


def gator(candles: np.ndarray,source_type: str = "close",sequential: bool = False):
    """
    expanded view of williams aligator only using source as input 
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_gator(source)

    return res if sequential else res[-1]