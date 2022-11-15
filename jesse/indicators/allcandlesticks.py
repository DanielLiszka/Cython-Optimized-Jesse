import numpy as np
import pandas_ta as ta 
import pandas as pd

from jesse.helpers import slice_candles,get_candle_source


def allcandlesticks(candles: np.ndarray,source_type: str = "close",sequential: bool = False):
    """
    returns columns containing all candlestick patterns
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)
    # source = get_candle_source(candles, source_type=source_type)
    high = pd.Series(candles[:,3])
    low = pd.Series(candles[:,4])
    open = pd.Series(candles[:,1])
    close = pd.Series(candles[:,2])
    df = pd.DataFrame()
    df = df.ta.cdl_pattern(open,high,low,close,name="all")
    res = df.to_numpy()

    return res if sequential else res[-1]