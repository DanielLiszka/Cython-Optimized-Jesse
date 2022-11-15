from typing import Union

import numpy as np
import tulipy as ti

from jesse.helpers import slice_candles


def ad(candles: np.ndarray, sequential: bool = False) -> Union[float, np.ndarray]:
    """
    AD - Chaikin A/D Line

    :param candles: np.ndarray
    :param sequential: bool - default: False

    :return: float | np.ndarray
    """
    candles = slice_candles(candles, sequential)

    res = ti.ad(np.ascontiguousarray(candles[:, 3]), np.ascontiguousarray(candles[:, 4]), np.ascontiguousarray(candles[:, 2]), np.ascontiguousarray(candles[:, 5]))

    return res if sequential else res[-1]