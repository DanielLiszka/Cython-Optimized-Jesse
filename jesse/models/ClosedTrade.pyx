# import numpy as np

from jesse.config import config
import random
# def uuid4():
  # s = '%032x' % random.getrandbits(128)
  # return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
from libc.math cimport abs, NAN
import jesse.helpers as jh
from jesse.config import config
from libc.math cimport abs, NAN
from jesse.enums import trade_types
import numpy as np 
cimport numpy as np 
cimport cython
from numpy cimport ndarray as ar 
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import ruuid as uuid

cdef inline double cython_sum(double[:] y) noexcept nogil:   
    cdef Py_ssize_t N = y.shape[0]
    cdef double x = y[0]
    cdef Py_ssize_t i
    for i in range(1,N):
        x += y[i]
    return x
    
class ClosedTrade():
    """A trade is made when a position is opened AND closed."""

    id: str = uuid.uuid4() # peewee.UUIDField(primary_key=True)
    strategy_name: str #peewee.CharField()
    symbol:str  # peewee.CharField()
    exchange: str #peewee.CharField()
    type: str #peewee.CharField()
    timeframe: str #peewee.CharField()
    opened_at: int = None # = peewee.BigIntegerField()
    closed_at: int = None #peewee.BigIntegerField()
    leverage: int = None #peewee.IntegerField()

    # class Meta:

        # database = database.db
        # indexes = ((('strategy_name', 'exchange', 'symbol'), False),)

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # peewee.Model.__init__(self, attributes=attributes, **kwargs)

        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)
            
        # used for fast calculation of the total qty, entry_price, exit_price, etc.
        self.buy_orders = DynamicNumpyArray((10, 2))
        self.sell_orders = DynamicNumpyArray((10, 2))
        # to store the actual order objects
        self.orders = []
        
    @property
    def to_json(self) -> dict:
        return {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "type": self.type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "fee": self.fee,
            "size": self.size,
            "PNL": self.pnl,
            "PNL_percentage": self.pnl_percentage,
            "holding_period": self.holding_period,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
        }
        
    @property
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'type': self.type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'qty': self.qty,
            'opened_at': self.opened_at,
            'closed_at': self.closed_at,
            "fee": self.fee,
            "size": self.size,
            "PNL": self.pnl,
            "PNL_percentage": self.pnl_percentage,
            "holding_period": self.holding_period,
        }

    @property
    def fee(self) -> float:
        cdef float trading_fee = config['env']['exchanges'][self.exchange]['fee']
        return trading_fee * self.qty * (self.entry_price + self.exit_price)

    @property
    def size(self) -> float:
        return self.qty * self.entry_price

    @property
    def pnl(self) -> float:
        """PNL"""
        fee = config['env']['exchanges'][self.exchange]['fee']
        return jh.estimate_PNL(
            self.qty, self.entry_price, self.exit_price,
            self.type, fee
        )

    @property
    def pnl_percentage(self) -> float:
        """
        Alias for self.roi
        """
        return self.roi

    @property
    def roi(self) -> float:
        """
        Return on Investment in percentage
        More at: https://www.binance.com/en/support/faq/5b9ad93cb4854f5990b9fb97c03cfbeb
        """
        return self.pnl / self.total_cost * 100

    @property
    def total_cost(self) -> float:
        """
        How much we paid to open this position (currently does not include fees, should we?!)
        """
        return self.entry_price * abs(self.qty) / self.leverage

    @property
    def holding_period(self):
        """How many SECONDS has it taken for the trade to be done."""
        return (self.closed_at - self.opened_at) / 1000
        
    @property
    def is_long(self) -> bool:
        return self.type == trade_types.LONG

    @property
    def is_short(self) -> bool:
        return self.type == trade_types.SHORT

    @property
    def qty(self) -> float:
        if self.type == trade_types.LONG:
            return cython_sum(self.buy_orders.array[:, 0])
        elif self.type == trade_types.SHORT:
            return cython_sum(self.sell_orders.array[:, 0])
        else:
            return 0.0

    @property
    def entry_price(self) -> float:
        if self.type == trade_types.LONG:
            orders = self.buy_orders.array  
        elif self.type == trade_types.SHORT:
            orders = self.sell_orders.array
        else:
            return NAN

        return cython_sum((orders[:, 0] * orders[:, 1])) / cython_sum(orders[:, 0])
        # return (orders[:, 0] * orders[:, 1]).sum() / orders[:, 0].sum()
    @property
    def exit_price(self) -> float:
        if self.type == trade_types.LONG:
            orders = self.sell_orders.array
        elif self.type == trade_types.SHORT:   
            orders = self.buy_orders.array
        else:
            return NAN

        return cython_sum((orders[:, 0] * orders[:, 1])) / cython_sum(orders[:, 0])
        # return (orders[:, 0] * orders[:, 1]).sum() / orders[:, 0].sum()


    @property
    def is_open(self) -> bool:
        return self.opened_at is not None
    
class DynamicNumpyArray:
    def __init__(self, shape: tuple,int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=DTYPE)
        self.bucket_size = shape[0]
        self.shape = shape

    def __len__(self) -> int:
        return self.index + 1
     
    def getslice(self,int start = 0, int stop =0):
        stop = self.index+1 if stop == 0 else stop 
        return self.array[start:stop] 
        
    def __setitem__(self, int i, ar item):
        self.array[self.index] = item
        
    def append(self, ar item) -> None:
        self.index += 1
        cdef ar new_bucket
        cdef Py_ssize_t index = self.index 
        cdef int bucket_size = self.bucket_size
        if index != 0 and (index + 1) % bucket_size == 0:
            new_bucket = np.zeros((self.shape),dtype=DTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=DTYPE)
        self.array[index] = item
        
    def __getitem__(self, i):
        stop = self.index + 1
        return self.array[i.start:stop]
        
