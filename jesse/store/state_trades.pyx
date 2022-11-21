
from typing import List
cimport cython
import numpy as np 
cimport numpy as np 
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import jesse.helpers as jh
from jesse.config import config
# from jesse.libs import DynamicNumpyArray
from jesse.models import store_trade_into_db
from jesse.models.Trade import Trade
from libc.math cimport abs, fmin
from numpy cimport ndarray as ar 
np.import_array()
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t
ctypedef double dtype_t
from jesse.services import selectors

cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float[:,::1] c_np_shift(float[:,::1] arr, int num, int fill_value=0):
    cdef float[:,::1] result = np.empty_like(arr)
    result[num:] = fill_value
    result[:num] = arr[-num:]
    return result
    
class TradesState:
    def __init__(self) -> None:
        self.storage = {}
        self.temp_storage = {}

    def init_storage(self) -> None:
        for ar in selectors.get_all_routes():
            exchange, symbol = ar['exchange'], ar['symbol']
            key = jh.key(exchange, symbol)
            self.storage[key] = DynamicNumpyArray((60, 6), drop_at=120)
            self.temp_storage[key] = DynamicNumpyArray((100, 4))

    def add_trade(self, trade: np.ndarray, exchange: str, symbol: str) -> None:
        key = f'{exchange}-{symbol}'
        if (
            len(self.temp_storage[key])
            and trade[0] - self.temp_storage[key].array[self.temp_storage[key].index][0] >= 1000
        ):
            arr = self.temp_storage[key]
            buy_arr = np.array(list(filter(lambda x: x[3] == 1, arr.array[0:arr.index+1])))
            sell_arr = np.array(list(filter(lambda x: x[3] == 0, arr.array[0:arr.index+1])))

            generated = np.array([
                # timestamp
                arr.array[arr.index][0],
                # price (weighted average)
                (arr.array[0:arr.index+1][:,1] * arr.array[0:arr.index+1][:,2]).sum() / arr.array[0:arr.index+1][:,2].sum(),
                # buy_qty
                0 if not len(buy_arr) else buy_arr[:, 2].sum(),
                # sell_qty
                0 if not len(sell_arr) else sell_arr[:, 2].sum(),
                # buy_count
                len(buy_arr),
                # sell_count
                len(sell_arr)
            ])

            # if jh.is_collecting_data():
                # store_trade_into_db(exchange, symbol, generated)
            # else:
            self.storage[key].append(generated)

            self.temp_storage[key].flush()
        self.temp_storage[key].append(trade)

    def get_trades(self, exchange: str, symbol: str) -> List[Trade]:
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[0:self.storage[key].index+1]

    @cython.wraparound(True)
    def get_current_trade(self, exchange: str, symbol: str) -> Trade:
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[self.storage[key].index]

    def get_past_trade(self, exchange: str, symbol: str, number_of_trades_ago: int) -> Trade:
        if number_of_trades_ago > 120:
            raise ValueError('Max accepted value for number_of_trades_ago is 120')

        number_of_trades_ago = abs(number_of_trades_ago)
        key = f'{exchange}-{symbol}'
        return self.storage[key].array[((self.storage[key].index+1)- abs(-1-number_of_trades_ago))] #[-1 - number_of_trades_ago]

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