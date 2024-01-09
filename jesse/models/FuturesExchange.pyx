
import numpy as np
cimport numpy as np 
DTYPE = np.float64
np.import_array()
cimport cython
from numpy cimport ndarray as ar 
import jesse.helpers as jh
# import jesse.services.logger as logger
from jesse.enums import sides, order_types
from jesse.exceptions import InsufficientMargin 
from jesse.models import Order
from jesse.services import selectors
from jesse.models.Exchange import Exchange
from libc.math cimport abs, fmax

class FuturesExchange(Exchange):

    def __init__(
            self,
            name: str,
            starting_balance: float,
            fee_rate: float,
            futures_leverage_mode: str,
            futures_leverage: int
    ):
        super().__init__(name, starting_balance, fee_rate, 'futures')

        self.futures_leverage_mode = futures_leverage_mode
        self.futures_leverage = futures_leverage
    @property
    def started_balance(self) -> float:
        return self.starting_assets[jh.app_currency()]
        
    @property
    def wallet_balance(self) -> float:
        return self.assets[self.settlement_currency]

    @property
    def available_margin(self) -> float:
        from jesse.store import store 
        # a temp which gets added to per each asset (remember that all future assets use the same currency for settlement)
        
        #cdef double temp_credits = self.assets[self.settlement_currency]
        cdef double margin = self.wallet_balance
        cdef double sum_buy_orders, sum_sell_orders
        cdef int c_leverage = self.futures_leverage
        # Calculate the total spent amount considering leverage
        # Here we need to calculate the total cost of all open positions and orders, considering leverage
        cdef float total_spent = 0.0
        for asset in self.assets:
            if asset == self.settlement_currency:
                continue
                
            key = f'{self.name}-{f"{asset}-{self.settlement_currency}"}'
            position = store.positions.storage.get(key, None)

            if position.qty != 0 and position:
                # Adding the cost of open positions
                total_spent += position.total_cost
                # add unrealized PNL
                total_spent -= position.pnl

            # Summing up the cost of open orders (buy and sell), considering leverage
            sum_buy_orders = (self.buy_orders[asset].array[0:self.buy_orders[asset].index+1][:,0] * self.buy_orders[asset].array[0:self.buy_orders[asset].index+1][:,1]).sum()
            sum_sell_orders = (self.sell_orders[asset].array[0:self.sell_orders[asset].index+1][:,0] * self.sell_orders[asset].array[0:self.sell_orders[asset].index+1][:,1]).sum()

            total_spent += fmax(
                abs(sum_buy_orders) / c_leverage, abs(sum_sell_orders) / c_leverage
            )

        # Subtracting the total spent from the margin
        margin -= total_spent

        return margin

    def charge_fee(self, double amount) -> None:
        cdef double c_fee_rate = self.fee_rate
        cdef double fee_amount = abs(amount) * c_fee_rate
        cdef double new_balance = self.assets[self.settlement_currency] - fee_amount
        # if fee_amount != 0:
            # logger.info(
                # f'Charged {round(fee_amount, 2)} as fee. Balance for {self.settlement_currency} on {self.name} changed from {round(self.assets[self.settlement_currency], 2)} to {round(new_balance, 2)}'
            # )
        self.assets[self.settlement_currency] = new_balance

    def add_realized_pnl(self, double realized_pnl) -> None:
        cdef double new_balance = self.assets[self.settlement_currency] + realized_pnl
        # logger.info(
            # f'Added realized PNL of {round(realized_pnl, 2)}. Balance for {self.settlement_currency} on {self.name} changed from {round(self.assets[self.settlement_currency], 2)} to {round(new_balance, 2)}')
        self.assets[self.settlement_currency] = new_balance

    def on_order_submission(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        cdef double order_size, remaining_margin, effective_order_size
        cdef double c_qty = order.qty 
        cdef double c_price = order.price

        # make sure we don't spend more than we're allowed considering current allowed leverage
        if not order.reduce_only:
            # Calculate the effective order size considering leverage
            effective_order_size = abs(order.qty * order.price) / self.futures_leverage
            if effective_order_size > self.available_margin:
                raise InsufficientMargin(
                    f'You cannot submit an order for ${round(order.qty * order.price)} when your effective margin balance is ${round(self.available_margin)} considering leverage')

        self.available_assets[base_asset] += c_qty

        if not order.reduce_only:
            if order.side == sides.BUY:
                self.buy_orders[base_asset].append(np.array([c_qty, c_price]))
            else:
                self.sell_orders[base_asset].append(np.array([c_qty, c_price]))

    def on_order_execution(self, order: Order) -> None:

        base_asset = order.symbol.split('-')[0]
        cdef Py_ssize_t index,counter
        cdef double c_qty = order.qty 
        cdef double c_price = order.price 
        
        if not order.reduce_only:
            if order.side == sides.BUY:
                # index = arr_equal(c_qty, c_price, self.buy_orders[base_asset].array)
                # self.buy_orders[base_asset][(index)] = np.array([0, 0])
                # find and set order to [0, 0] (same as removing it)
                for index in reversed(range(0,(self.buy_orders[base_asset].index+1))):
                    item = self.buy_orders[base_asset].array[index]
                    if item[0] == c_qty and item[1] == c_price:
                        self.buy_orders[base_asset][(index)] = np.array([0, 0])
                        break
            else:
                # find and set order to [0, 0] (same as removing it)
                for index in reversed(range(0,(self.sell_orders[base_asset].index+1))):
                    item = self.sell_orders[base_asset].array[index]
                    if item[0] == c_qty and item[1] == c_price:
                        self.sell_orders[base_asset][index] = np.array([0, 0])
                        break


    def on_order_cancellation(self, order: Order) -> None:
        base_asset = order.symbol.split('-')[0]
        cdef Py_ssize_t index,counter
        cdef double c_qty = order.qty 
        cdef double c_price = order.price 
        self.available_assets[base_asset] -= c_qty
        # self.available_assets[quote_asset] += order.qty * order.price
        if not order.reduce_only:
            if order.side == sides.BUY:
                # find and set order to [0, 0] (same as removing it)
                for index in reversed(range(0,(self.buy_orders[base_asset].index+1))):
                    item = self.buy_orders[base_asset].array[index]
                    if item[0] == c_qty and item[1] == c_price:
                        self.buy_orders[base_asset][index] = np.array([0, 0])
                        break
            else:
                # find and set order to [0, 0] (same as removing it)
                for index in reversed(range(0,(self.sell_orders[base_asset].index+1))):
                    item = self.sell_orders[base_asset].array[index]
                    if item[0] == c_qty and item[1] == c_price:
                        self.sell_orders[base_asset][index] = np.array([0, 0])
                        break
                        

class DynamicNumpyArray:
    def __init__(self, shape: tuple,int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=DTYPE)
        self.bucket_size = shape[0]
        self.shape = shape

    def __len__(self) -> int:
        # cdef Py_ssize_t index = self.index 
        return self.index + 1
             
    def __setitem__(self, int i, ar item):
        cdef Py_ssize_t index = self.index
        if i < 0:
            i = (index + 1) - abs(i)
        self.array[i] = item
        
    def append(self, ar item) -> None:
        self.index += 1
        cdef ar new_bucket
        cdef Py_ssize_t index = self.index 
        cdef int bucket_size = self.bucket_size
        # expand if the arr is almost full
        if index != 0 and (index + 1) % bucket_size == 0:
            new_bucket = np.zeros((self.shape),dtype=DTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=DTYPE)
        self.array[index] = item
