from typing import Union

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def lag(candles: np.ndarray, seriesA: np.ndarray, length:int=5, sequential: bool = False) -> Union[float, np.ndarray]:
    """
    lag of one input series by period
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    # candles = slice_candles(candles, sequential)

    res = ti.lag(np.ascontiguousarray(seriesA),length)

    return res if sequential else res[-1]