import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def vhf(candles: np.ndarray, length:int=5, sequential: bool = False):
    """
    vertical horizontal filter
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)

    res = ti.vhf(np.ascontiguousarray(seriesA),length)

    return res if sequential else res[-1]