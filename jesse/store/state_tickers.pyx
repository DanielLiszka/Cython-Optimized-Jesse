
from typing import List
import numpy as np
cimport numpy as np 
np.import_array()
cimport cython
import jesse.helpers as jh
from jesse.services import selectors
from jesse.libs import DynamicNumpyArray
from jesse.models import store_ticker_into_db, Ticker
from libc.math cimport abs, fmin
from numpy cimport ndarray as ar 
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t
ctypedef double dtype_t

cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float[:,::1] c_np_shift(float[:,::1] arr, int num, int fill_value=0):
    cdef float[:,::1] result = np.empty_like(arr)
    result[num:] = fill_value
    result[:num] = arr[-num:]
    return result

class TickersState:
    def __init__(self) -> None:
        self.storage = {}

    def init_storage(self) -> None:
        for ar in selectors.get_all_routes():
            exchange, symbol = ar['exchange'], ar['symbol']
            key = jh.key(exchange, symbol)  
            self.storage[key] = DynamicNumpyArray((60, 5), drop_at=120)

    @cython.wraparound(True)
    def add_ticker(self, ticker: np.ndarray, exchange: str, symbol: str) -> None:
        key = f'{exchange}-{symbol}'

        # only process once per second
        if len(self.storage[key]) == 0 or jh.now_to_timestamp() - self.storage[key].array[self.storage[key].index][0] >= 1000:#[-1][0] >= 1000:
            self.storage[key].append(ticker)

            # if jh.is_collecting_data():
                # store_ticker_into_db(exchange, symbol, ticker)
                # return

    def get_tickers(self, exchange: str, symbol: str) -> np.ndarray[Ticker]:
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[0:self.storage[key].index+1]


    @cython.wraparound(True)
    def get_current_ticker(self, exchange: str, symbol: str) -> Ticker:
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[self.storage[key].index]

    def get_past_ticker(self, exchange: str, symbol: str, int number_of_tickers_ago) -> Ticker:
        if number_of_tickers_ago > 120:
            raise ValueError('Max accepted value for number_of_tickers_ago is 120')

        number_of_tickers_ago = abs(number_of_tickers_ago)
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[((self.storage[key].index+1)- abs(-1-number_of_tickers_ago))] #[-1 - number_of_tickers_ago]

class DynamicNumpyArray:
    
    def __init__(self, shape: tuple, int drop_at=-1, int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=FTYPE)
        self.bucket_size = shape[0]
        self.shape = shape
        self.drop_at = drop_at

    def __len__(self) -> int:
        return self.index + 1
        
    def __setitem__(self, int i, ar item) -> None:
        if i < 0:
            i = (self.index + 1) - abs(i)
        self.array[i] = item
        
    def append(self, ar item) -> None:
        self.index += 1
        cdef int shift_num
        cdef ar new_bucket, result
        cdef Py_ssize_t index = self.index 
        cdef int bucket_size = self.bucket_size
        cdef int drop_at = self.drop_at
        
        # expand if the arr is almost full
        if index != 0 and (index + 1) % bucket_size == 0:
            new_bucket = np.zeros(self.shape,dtype=FTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=FTYPE)

        # drop N% of the beginning values to free memory
        if (
            drop_at is not -1
            and index != 0
            and (index + 1) % drop_at == 0
        ):
            shift_num = int(drop_at / 2)
            self.index -= shift_num            
            self.array = c_np_shift(self.array, -shift_num)

        self.array[self.index] = item

    def flush(self) -> None:
        self.index = -1
        self.array = np.zeros(self.shape,dtype=FTYPE)
        self.bucket_size = self.shape[0]