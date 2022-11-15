from stock_indicators import indicators

import numpy as np

from jesse.helpers import slice_candles,get_candle_source


def connors_rsi(candles: np.ndarray, rsi_period:int=3, streak_period:int=2,rank_period:int=100,source_type: str = "close",sequential: bool = False):
    """
    Connor's RSI
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    res = indicators.get_get_connors_rsi(source,rsi_period,streak_period,rank_period)

    return res if sequential else res[-1]