from typing import Union

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def crossover(candles: np.ndarray, seriesA: np.ndarray, seriesB: np.ndarray, sequential: bool = False) -> Union[float, np.ndarray]:
    """
    crossover of two series inputs
    1 if a > b and a-1 < b-1 else 0 
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    # candles = slice_candles(candles, sequential)

    res = ti.crossover(np.ascontiguousarray(seriesA),np.ascontiguousarray(seriesB))

    return res if sequential else res[-1]