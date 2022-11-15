from typing import Union

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def lineardecay(candles: np.ndarray, seriesA: np.ndarray, length:int=5, sequential: bool = False) -> Union[float, np.ndarray]:
    """
    linear decay of input series over period
    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    # candles = slice_candles(candles, sequential)

    res = ti.decay(np.ascontiguousarray(seriesA),length)

    return res if sequential else res[-1]