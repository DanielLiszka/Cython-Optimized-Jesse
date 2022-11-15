import numpy as np
import pandas_ta as ta 
import pandas as pd

from jesse.helpers import slice_candles,get_candle_source


def linereg_offset(candles: np.ndarray, indicator_source: np.ndarray = None,length:int=14,offset:int=0,source_type: str = "close",sequential: bool = False):
    """
    linear regression line with offset
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    series = pd.Series(source) if indicator_source is None else pd.Series(indicator_source)
    df = pd.DataFrame()
    df = df.ta.linereg(close=series,length=length,offset=offset)
    res = df.to_numpy()

    return res if sequential else res[-1]