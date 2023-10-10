from typing import Union
import sys
import random
# def uuid4():
  # s = '%032x' % random.getrandbits(128)
  # return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
import numpy as np
cimport numpy as np 
np.import_array()
import jesse.helpers as jh
import jesse.services.selectors as selectors
from jesse.enums import trade_types, order_types
from jesse.exceptions import EmptyPosition, OpenPositionError
from jesse.models import Order, Exchange
# from jesse.services import logger
from jesse.utils import sum_floats, subtract_floats
from libc.math cimport abs,NAN
import ruuid as uuid

class Position:
    def __init__(self, exchange_name: str, symbol: str, attributes: dict = None) -> None:
        self.id = uuid.uuid4()
        self.entry_price: float = None
        self.exit_price: float = None
        self.current_price: float = None
        self.qty: float = 0
        self.previous_qty: float = 0
        self.opened_at = None
        self.closed_at = None
        self._mark_price: float = None
        self._funding_rate: float = None
        self._next_funding_timestamp = None
        self._liquidation_price: float = None

        if attributes is None:
            attributes = {}
        from jesse.store import store 
        self.exchange_name = exchange_name
        self.exchange: Exchange = store.exchanges.storage.get(self.exchange_name, None)

        self.symbol = symbol
        self.strategy = None

        for a in attributes:
            setattr(self, a, attributes[a])

    @property
    def mark_price(self) -> float:
        return self.current_price

    @property
    def funding_rate(self) -> float:
        return 0 

    @property
    def next_funding_timestamp(self) -> Union[int, None]:
        return None


    @property
    def value(self) -> float:
        """
        The value of open position in the quote currency

        :return: float
        """
        if self.type == 'close':
            return 0
            
        if self.current_price is None:
            return None
            
        return abs(self.current_price * self.qty)

    @property
    def type(self) -> str:
        """
        The type of open position - long, short, or close

        :return: str
        """
        c_qty = self.qty
        if c_qty > 0:
            return 'long'
        elif c_qty < 0:
            return 'short'

        return 'close'

    @property
    def pnl_percentage(self) -> float:
        """
        Alias for self.roi

        :return: float
        """
        return self.roi

    @property
    def roi(self) -> float:
        """
        Return on Investment in percentage
        More at: https://www.binance.com/en/support/faq/5b9ad93cb4854f5990b9fb97c03cfbeb
        """
        if self.pnl == 0:
            return 0
            
        return self.pnl / self.total_cost * 100

    @property
    def total_cost(self) -> float:
        """
        How much we paid to open this position (currently does not include fees, should we?!)
        """
        cdef double c_qty = self.qty
        cdef c_entry_price = self.entry_price
        if c_qty == 0:
            return NAN

        cdef double base_cost = c_entry_price * abs(c_qty)
        if self.strategy:
            return base_cost / self.leverage

        return base_cost

    @property
    def leverage(self) -> double:
        if self.exchange_type == 'spot':
            return 1

        if self.strategy:
            return self.strategy.leverage
        else:
            return NAN
            
    @property
    def exchange_type(self) -> str:
        return self.exchange.type
        
    @property
    def entry_margin(self) -> float:
        """
        Alias for self.total_cost
        """
        return self.total_cost

    @property
    def pnl(self) -> float:
        """
        The PNL of the position

        :return: float
        """
        cdef double c_qty = self.qty
        if abs(c_qty) < self._min_qty:
            return 0
            
        if self.entry_price is None:
            return 0
            
        if self.value is None:
            return 0
            
        diff = abs(self.current_price * self.qty) - abs(self.entry_price * c_qty)

        return -diff if self.qty < 0 else diff

    @property
    def is_open(self) -> bool:
        """
        Is the current position open?

        :return: bool
        """
        return self.type in ['long', 'short']

    @property
    def is_close(self) -> bool:
        """
        Is the current position close?

        :return: bool
        """
        return self.type == 'close'
    @property
    def is_long(self) -> bool:
        return self.qty > self._min_qty
        
    @property
    def is_short(self) -> bool: 
        return self.qty < -abs(self.min_qty)
        
    @property
    def mode(self) -> str:
        if self.exchange.type == 'spot':
            return 'spot'
        else:
            return self.exchange.futures_leverage_mode

    @property
    def liquidation_price(self) -> double:
        """
        The price at which the position gets liquidated. formulas are taken from:
        https://help.bybit.com/hc/en-us/articles/900000181046-Liquidation-Price-USDT-Contract-
        """
        cdef double c_qty = self.qty
        if c_qty == 0:
            return NAN

        if self.exchange.type == 'spot' or self.exchange.futures_leverage_mode == 'cross':
            return NAN

        else: #elif self.mode == 'isolated':
            if c_qty > 0:    
                return self.entry_price * (1 - (1 / self.strategy.leverage) + 0.004)
            elif c_qty < 0:
                return self.entry_price * (1 + (1 / self.strategy.leverage) - 0.004)
            else:
                return NAN

        # else:
            # raise ValueError

    @property
    def _initial_margin_rate(self) -> float:
        return 1 / self.leverage

    @property
    def bankruptcy_price(self) -> double:
        cdef double c_qty = self.qty
        if c_qty > 0:
            return self.entry_price * (1 - 1 / self.leverage)
        elif c_qty < 0:
            return self.entry_price * (1 + 1 / self.leverage)
        else:
            return NAN
            
    @property
    def to_dict(self):
        return {
            'entry_price': self.entry_price,
            'qty': self.qty,
            'current_price': self.current_price,
            'value': self.value,
            'type': self.type,
            'exchange': self.exchange_name,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'leverage': self.leverage,
            'liquidation_price': self.liquidation_price,
            'bankruptcy_price': self.bankruptcy_price,
            'mode': self.mode,
        }
        
    def _mutating_close(self, close_price: float) -> None:
        cdef double c_qty = self.qty
        if self.type == 'close' and self._can_mutate_qty:
            raise EmptyPosition('The position is already closed.')

        self.exit_price = close_price
        self.closed_at = jh.now_to_timestamp()
        
        if self.exchange and self.exchange.type == 'futures':
            # just to prevent confusion
            close_qty = abs(c_qty)
            estimated_profit = jh.estimate_PNL(
                close_qty, self.entry_price,
                close_price, self.type
            )
            self.exchange.add_realized_pnl(estimated_profit)
            self.exchange.temp_reduced_amount[self.symbol.split('-')[0]] += abs(close_qty * close_price)

        if self._can_mutate_qty:
            self._update_qty(0, operation='set')
        self.entry_price = None
        self._close()

    def _close(self):
        from jesse.store import store
        store.completed_trades.close_trade(self)

    def _mutating_reduce(self, qty: float, price: float) -> None:
        if not self._can_mutate_qty:
            return
        cdef double c_qty = self.qty
        if c_qty != 0 is False:
            raise EmptyPosition('The position is closed.')

        # just to prevent confusion
        qty = abs(qty)

        estimated_profit = jh.estimate_PNL(qty, self.entry_price, price, self.type)

        if self.exchange and self.exchange.type == 'futures':
            # self.exchange.increase_futures_balance(qty * self.entry_price + estimated_profit)
            self.exchange.add_realized_pnl(estimated_profit)
            self.exchange.temp_reduced_amount[self.symbol.split('-')[0]] += abs(qty * price)

        if c_qty > 0: #self.type == trade_types.LONG:
            self._update_qty(qty, operation='subtract')
        elif c_qty < 0: #self.type == trade_types.SHORT:
            self._update_qty(qty, operation='add')

    def _mutating_increase(self, qty: float, price: float) -> None:
        cdef double c_qty = self.qty
        if not c_qty != 0:
            raise OpenPositionError('position must be already open in order to increase its size')

        qty = abs(qty)

        self.entry_price = jh.estimate_average_price(
            qty, price, self.qty,
            self.entry_price
        )


        if self._can_mutate_qty:
            if self.type == trade_types.LONG:
                self._update_qty(qty, operation='add')
            elif self.type == trade_types.SHORT:
                self._update_qty(qty, operation='subtract')

    def _mutating_open(self, qty: float, price: float) -> None:
        cdef double c_qty = self.qty
        if self.type != 'close' and self._can_mutate_qty:
            raise OpenPositionError('an already open position cannot be opened')

        self.entry_price = price
        self.exit_price = None
        if self._can_mutate_qty:
            self._update_qty(qty, operation='set')
        self.opened_at = jh.now_to_timestamp()

        self._open()
        
    def _update_qty(self, qty: float, operation='set'):
        self.previous_qty = self.qty

        if self.exchange_type == 'spot':
            if operation == 'set':
                self.qty = qty * (1 - self.exchange.fee_rate)
            elif operation == 'add':
                self.qty = sum_floats(self.qty, qty * (1 - self.exchange.fee_rate))
            elif operation == 'subtract':
                # fees are taken from the quote currency. in spot mode, sell orders cause
                # the qty to reduce but fees are handled on the exchange balance stuff
                self.qty = subtract_floats(self.qty, qty)

        elif self.exchange_type == 'futures':
            if operation == 'set':
                self.qty = qty
            elif operation == 'add':
                self.qty = sum_floats(self.qty, qty)
            elif operation == 'subtract':
                self.qty = subtract_floats(self.qty, qty)
        else:
            raise NotImplementedError('exchange type not implemented')
            
    def _open(self):
        from jesse.store import store
        store.completed_trades.open_trade(self)

    def _on_executed_order(self, order: Order) -> None:
        cdef double qty = order.qty
        cdef double price = order.price
        cdef double c_qty = self.qty
        cdef double diff_qty
        qty = order.qty
        price = order.price

        # if jh.is_livetrading():
            # if order.is_partially_filled:
                # before_qty = self.qty - order.filled_qty
            # else:
                # before_qty = self.qty - order.qty
            # after_qty = self.qty
            # if before_qty != 0 and after_qty == 0:
                # self._close()
        # else: 
        
        # TODO: detect reduce_only order, and if so, see if you need to adjust qty and price (above variables)

        if self.exchange and self.exchange.type == 'futures':
            self.exchange.charge_fee(qty * price)

        # order opens position
        if self.qty == 0:
            change_balance = order.type == order_types.MARKET
            self._mutating_open(qty, price)
        # order closes position
        elif (sum_floats(self.qty, qty)) == 0:
            self._mutating_close(price)
        # order increases the size of the position
        elif self.qty * qty > 0:
        
            if order.reduce_only:
                # logger.info('Did not increase position because order is a reduce_only order')
                self._close(price)
            else:
                self._mutating_increase(qty, price)
        # order reduces the size of the position
        elif self.qty * qty < 0:
            # if size of the order is big enough to both close the
            # position AND open it on the opposite side
            if abs(qty) > abs(self.qty):
                if order.reduce_only:
                    # logger.info(
                        # f'Executed order is bigger than the current position size but it is a reduce_only order so it just closes it. Order QTY: {qty}, Position QTY: {self.qty}')
                    self._mutating_close(price)
                else:
                    # logger.info(
                        # f'Executed order is big enough to not close, but flip the position type. Order QTY: {qty}, Position QTY: {self.qty}')
                    diff_qty = sum_floats(self.qty, qty)
                    self._mutating_close(price)
                    self._mutating_open(diff_qty, price)
            else:
                self._mutating_reduce(qty, price)

        if self.strategy:
            self.strategy._on_updated_position(order)

    def update_from_stream(self, data: dict, is_initial: bool) -> None:
        """
        Used for updating the position from the WS stream (only for live trading)
        """
        if self.exchange_type == 'futures':
            self.entry_price = data['entry_price']
            self._liquidation_price = data['liquidation_price']
        # if the new qty (data['qty']) is different than the current (self.qty) then update it:
        if self.qty != data['qty']:
            self.previous_qty = self.qty
            self.qty = data['qty']

        before_qty = abs(self.qty)
        after_qty = abs(data['qty'])
        # if opening position
        if before_qty == 0 and after_qty != 0:
            if is_initial:
                from jesse.store import store
                store.completed_trades.add_order_record_only(
                    self.exchange_name, self.symbol, jh.type_to_side(self.type),
                    self.qty, self.entry_price
                )
            self.opened_at = jh.now_to_timestamp()
            self._open()
        # if closing position
        elif before_qty != 0 and after_qty == 0:
            self.closed_at = jh.now_to_timestamp()
            
    @property
    def _min_notional_size(self) -> float:
        if not (jh.is_livetrading() and self.exchange_type == 'spot'):
            return 0

        return self.exchange.vars['precisions'][self.symbol]['min_notional_size']

    @property
    def _min_qty(self) -> float:
        if not (jh.is_livetrading() and self.exchange_type == 'spot'):
            return 0
            
        min_notional_size = self._min_notional_size

        # few exchanges like FTX have min_qty instead of min_notional_size, hence:
        if min_notional_size is None:
            return self.exchange.vars['precisions'][self.symbol]['min_qty']
            
        return self._min_notional_size / self.current_price
        
    @property
    def _can_mutate_qty(self):
        return not (self.exchange_type == 'spot' and jh.is_livetrading())