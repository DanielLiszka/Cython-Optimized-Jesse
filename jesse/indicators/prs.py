from stock_indicators import indicators

import numpy as np

from jesse.helpers import slice_candles,get_candle_source


def prs(candles: np.ndarray, length:int=10,source_type: str = "close",sequential: bool = False):
    """
    price relative strength
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_prs(source,length1,length2,length3)

    return res if sequential else res[-1]