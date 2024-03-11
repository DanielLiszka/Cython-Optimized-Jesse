
import arrow
import numpy as np
cimport numpy as np 
cimport cython
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import jesse.helpers as jh
# from jesse.services import logger
from libc.math cimport isnan, NAN 
from numpy.math cimport INFINITY
from colorama import Fore, Style

@cython.wraparound(True)
def generate_candle_from_one_minutes(double [:,::1] array):  
    cdef Py_ssize_t rows, i 
    rows = array.shape[0]
    cdef double sum1 = 0.0
    cdef double min1 = INFINITY
    cdef double max1 = -INFINITY
    cdef double close1, open1, time1
    close1 = array[-1,2] if array[-1,2] == array[-1,2] else NAN
    open1 = array[0,1] if array[0,1] == array[0,1] else NAN
    time1 =  array[0,0]
    if close1 is not NAN: 
        for i in range(rows):
            sum1 = sum1 + array[i,5] 
            if array[i,4] < min1:
                min1 = array[i,4]
            if array[i,3] > max1:
                max1 = array[i,3] 
    else:
        sum1 = NAN
        min1 = NAN
        max1 = NAN
        
    return np.array([
        time1,
        open1,
        close1,
        max1,
        min1,
        sum1,
    ])
    
#old wrapper       
# @cython.wraparound(True)
# def generate_candle_from_one_minutes(timeframe: str,
                                     # candles,
                                     # bint accept_forming_candles = False):
    # if len(candles) == 0:
        # raise ValueError('No candles were passed')

    # if not accept_forming_candles and len(candles) != jh.timeframe_to_one_minutes(timeframe):
        # raise ValueError(
            # f'Sent only {len(candles)} candles but {jh.timeframe_to_one_minutes(timeframe)} is required to create a "{timeframe}" candle.'
        # )
    # sum1, min1, max1, close1, open1, time1 = c_sum(candles)
    # return np.array([
        # time1,
        # open1,
        # close1,
        # max1,
        # min1,
        # sum1,
    # ])
def candle_dict_to_np_array(candle: dict) -> np.ndarray:
    return np.array([
        candle['timestamp'],
        candle['open'],
        candle['close'],
        candle['high'],
        candle['low'],
        candle['volume']
    ])
    
@cython.wraparound(True)
def print_candle(candle: np.ndarray, is_partial: bool, symbol: str) -> None:
    """
    Ever since the new GUI dashboard, this function should log instead of actually printing

    :param candle: np.ndarray
    :param is_partial: bool
    :param symbol: str
    """
    if jh.should_execute_silently():
        return

    candle_form = '  ==' if is_partial else '===='
    candle_info = f' {symbol} | {str(arrow.get(candle[0] / 1000))[:-9]} | {candle[1]} | {candle[2]} | {candle[3]} | {candle[4]} | {round(candle[5], 2)}'
    msg = candle_form + candle_info

    # store it in the log file
    # logger.info(msg)


def is_bullish( candle: np.ndarray) -> bool:
    return candle[2] >= candle[1] 


def is_bearish( candle: np.ndarray) -> bool:
    return candle[2] < candle[1]   


def candle_includes_price( candle: np.ndarray, float price) -> bool:
    return (price >= candle[4]) and (price <= candle[3])


def split_candle(candle, price:float) -> tuple: 
    """
    splits a single candle into two candles: earlier + later

    :param candle: np.ndarray
    :param price: float

    :return: tuple
    """
    cdef DTYPE_t timestamp, o, c, h, l, v 

    timestamp = candle[0]
    o = candle[1]
    c = candle[2]
    h = candle[3]
    l = candle[4]
    v = candle[5]
    if (c >= o) and l < price < o:
        return np.array([
            timestamp, o, price, o, price, v
        ]), np.array([
            timestamp, price, c, h, l, v
        ])
    elif price == o:
        return candle, candle
    elif (c < o) and o < price < h:
        return np.array([
            timestamp, o, price, price, o, v
        ]), np.array([
            timestamp, price, c, h, l, v
        ])
    elif (c < o) and l < price < c:
        return np.array([
            timestamp, o, price, h, price, v
        ]), np.array([
            timestamp, price, c, c, l, v
        ])
    elif (c >= o) and c < price < h:
        return np.array([
            timestamp, o, price, price, l, v
        ]), np.array([
            timestamp, price, c, h, c, v
        ]),
    elif (c < o) and price == c:
        return np.array([
            timestamp, o, c, h, c, v
        ]), np.array([
            timestamp, price, price, price, l, v
        ])
    elif (c >= o) and price == c:
        return np.array([
            timestamp, o, c, c, l, v
        ]), np.array([
            timestamp, price, price, h, price, v
        ])
    elif (c < o) and price == h:
        return np.array([
            timestamp, o, h, h, o, v
        ]), np.array([
            timestamp, h, c, h, l, v
        ])
    elif (c >= o) and price == l:
        return np.array([
            timestamp, o, l, o, l, v
        ]), np.array([
            timestamp, l, c, h, l, v
        ])
    elif (c < o) and price == l:
        return np.array([
            timestamp, o, l, h, l, v
        ]), np.array([
            timestamp, l, c, c, l, v
        ])
    elif (c >= o) and price == h:
        return np.array([
            timestamp, o, h, h, l, v
        ]), np.array([
            timestamp, h, c, h, c, v
        ])
    elif (c < o) and c < price < o:
        return np.array([
            timestamp, o, price, h, price, v
        ]), np.array([
            timestamp, price, c, price, l, v
        ])
    elif (c >= o) and o < price < c:
        return np.array([
            timestamp, o, price, price, l, v
        ]), np.array([
            timestamp, price, c, h, price, v
        ])
    # else:
        # print(f'{Fore.RED} error {Style.RESET_ALL}')
        # print(f'candle: {candle} -  price {price}')
        # return candle, candle