from typing import Union

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def crossany(candles: np.ndarray, seriesA: np.ndarray, seriesB: np.ndarray, sequential: bool = False) -> Union[float, np.ndarray]:
    """
    crossany of two series inputs
    1 = cross  0 = no cross
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    # candles = slice_candles(candles, sequential)

    res = ti.crossany(np.ascontiguousarray(seriesA),np.ascontiguousarray(seriesB))

    return res if sequential else res[-1]