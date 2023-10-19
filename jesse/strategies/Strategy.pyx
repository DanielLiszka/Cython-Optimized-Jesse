from abc import ABC, abstractmethod
from time import sleep
from typing import List, Dict, Union
from enum import Enum
import sys
from jesse.config import config
import numpy as np
cimport numpy as np 
cimport cython
np.import_array()

from libc.math cimport NAN, fmaxf, fabsf

import random  
import jesse.helpers as jh
import jesse.services.logger as logger
import jesse.services.selectors as selectors
from jesse import exceptions
from jesse.enums import sides, order_submitted_via
from jesse.models import ClosedTrade, Order, Route, FuturesExchange, SpotExchange, Position
from jesse.services import metrics
from jesse.services.broker import Broker
from jesse.store import store
from jesse.services.cache import cached
from jesse.services import notifier
from jesse.enums import order_statuses
from jesse.routes import router
# from gym import spaces, Space
import ruuid as uuid

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t   

from cpython cimport * 
cdef extern from "Python.h":
        Py_ssize_t PyList_GET_SIZE(object list)
        PyObject* PyList_GET_ITEM(object list, Py_ssize_t i)
        void PyList_SET_ITEM(object list, Py_ssize_t i, object o)
        list PyList_New(Py_ssize_t len)       
        object PyObject_GetItem(object o, object key)
        object PyObject_GetAttr(object o, object attr_name)
        bint PyObject_RichCompareBool(object o1, object o2, int opid)
        int PyList_Append(object list, object item)
        object PyUnicode_RichCompare(object left, object right, int op)
        
        
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
#cython: np_pythran=True

cdef inline bint arr_equal(double [:,::1] a1, double [:,::1] a2) nogil noexcept:
    if a1.shape[0] != a2.shape[0]:
        return False
    cdef Py_ssize_t rows, columns
    rows = a1.shape[0]
    columns = a1.shape[1] 
    for i in reversed(range(rows)):
        for j in range(columns):
            if a1[i,j] != a2[i,j]:
                return False 
                break 
    return True
    
# def arr_equal(np.ndarray a1, np.ndarray a2):
    # return np.array_equal(a1,a2)
# def uuid4():
  # s = '%032x' % random.getrandbits(128)
  # return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
  
class Strategy(ABC):
    """
    The parent strategy class which every strategy must extend. It is the heart of the framework!
    """

    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.name = None
        self.symbol = None
        self.exchange = None
        self.timeframe = None
        self.hp = None 
        self.full_path_name = None
        self.full_name = None
        self.index:int  = 0
        self.vars = {}
        self.preload_candles = False

        self.increased_count: int = 0
        self.reduced_count: int = 0
        self.counter: int = 0 
        
        self.buy = None
        self._buy = None
        self.sell = None
        self._sell = None
        self.stop_loss = None
        self._stop_loss = None
        self.take_profit = None
        self._take_profit = None

        self._executed_open_orders:list = []
        self._executed_close_orders:list = []
        
        self._current_candle:np.ndarray = None
        self._indicator1_value_tf = None
        self._indicator2_value_tf = None
        self._indicator3_value_tf = None
        self._indicator4_value_tf = None
        self._indicator5_value_tf = None
        self._indicator6_value_tf = None
        self._indicator7_value_tf = None
        self._indicator8_value_tf = None
        self._indicator9_value_tf = None
        self._indicator10_value_tf = None
        self.other_tf: str = None
       
        self._indicator1_value = None
        self._indicator2_value = None 
        self._indicator3_value = None
        self._indicator4_value = None
        self._indicator5_value = None
        self._indicator6_value = None
        self._indicator7_value = None
        self._indicator8_value = None
        self._indicator9_value = None
        self._indicator10_value = None 
        
        
        self.trade: ClosedTrade = None
        self.trades_count: int = 0

        self._is_executing = False
        self._is_initiated = False
        self._is_handling_updated_order = False

        self.position: Position = None
        self.broker = None

        self._cached_methods:dict = {}
        self._cached_metrics:dict = {}
        self.slice_amount:dict = {}
        
    def update_new_candle(self, candle, exchange, symbol, timeframe):
        pass
        
    def _init_objects(self) -> None:
        """
        This method gets called after right creating the Strategy object. It
        is just a workaround as a part of not being able to set them inside
        self.__init__() for the purpose of removing __init__() methods from strategies.
        """
        key = f'{self.exchange}-{self.symbol}'
        self.position = store.positions.storage.get(key, None)
        self.broker = Broker(self.position, self.exchange, self.symbol, self.timeframe)
        if jh.get_config('env.simulation.preload_candles'):
            self.preload_candles = True
        if self.hp is None and len(self.hyperparameters()) > 0:
            self.hp = {}
            for dna in self.hyperparameters():
                self.hp[dna['name']] = dna['default']
                
        self.before_start()
    @property
    def _price_precision(self) -> int:
        """
        used when live trading because few exchanges require numbers to have a specific precision
        """
        return selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision']

    @property
    def _qty_precision(self) -> int:
        """
        used when live trading because few exchanges require numbers to have a specific precision
        """
        return selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['qty_precision']

    def _broadcast(self, msg: str) -> None:
        """Broadcasts the event to all OTHER strategies

        Arguments:
            msg {str} -- [the message to broadcast]
        """

        for r in router.routes:
            # skip self
            if r.strategy.id == self.id:
                continue

            if msg == 'route-open-position':
                r.strategy.on_route_open_position(self)
            elif msg == 'route-close-position':
                r.strategy.on_route_close_position(self)
            elif msg == 'route-increased-position':
                r.strategy.on_route_increased_position(self)
            elif msg == 'route-reduced-position':
                r.strategy.on_route_reduced_position(self)
            elif msg == 'route-canceled':
                r.strategy.on_route_canceled(self)

            r.strategy._detect_and_handle_entry_and_exit_modifications()

    def _on_updated_position(self, order: Order) -> None:
        """
        Handles the after-effect of the executed order

        Note that it assumes that the position has already been affected
        by the executed order.

        Arguments:
            order {Order} -- the executed order object
        """
        # in live-mode, sometimes order-update effects and new execution has overlaps, so:
        # self._is_handling_updated_order = True

        # this is the last executed order, and had its effect on
        # the position. We need to know what its effect was:
        before_qty = self.position.previous_qty
        after_qty = self.position.qty

        # call the relevant strategy event handler:
        # if opening position
        if fabsf(before_qty) <= fabsf(self.position._min_qty) < fabsf(after_qty):
            effect = 'opening_position'
        # if closing position
        elif fabsf(before_qty) > fabsf(self.position._min_qty) >= fabsf(after_qty):
            effect = 'closing_position'
        # if increasing position size
        elif fabsf(after_qty) > fabsf(before_qty):
            effect = 'increased_position'
        # if reducing position size
        else: # fabsf(after_qty) < fabsf(before_qty):
            effect = 'reduced_position'

        # call the relevant strategy event handler:
        if effect == 'opening_position':
            txt = f"OPENED {self.position.type} position for {self.symbol}: qty: {after_qty}, entry_price: {self.position.entry_price}"
            if jh.is_debuggable('position_opened'):
                logger.info(txt)
            self._on_open_position(order)
        # if closing position
        elif effect == 'closing_position':
            txt = f"CLOSED Position for {self.symbol}"
            if jh.is_debuggable('position_closed'):
                logger.info(txt)
            self._on_close_position(order)
        # if increasing position size
        elif effect == 'increased_position':
            txt = f"INCREASED Position size to {after_qty}"
            if jh.is_debuggable('position_increased'):
                logger.info(txt)
            self._on_increased_position(order)
        # if reducing position size
        else: # if effect == 'reduced_position':
            txt = f"REDUCED Position size to {after_qty}"
            if jh.is_debuggable('position_reduced'):
                logger.info(txt)
            self._on_reduced_position(order)
        self._is_handling_updated_order = False

    def filters(self) -> list:
        return []

    def hyperparameters(self) -> list:
        return []

    def dna(self) -> str:
        return ''

    def _execute_long(self) -> None:
        self.go_long()

        # validation
        if self.buy is None:
            raise exceptions.InvalidStrategy('You forgot to set self.buy. example (qty, price)')
        elif type(self.buy) not in [tuple, list]:
            raise exceptions.InvalidStrategy(
                f'self.buy must be either a list or a tuple. example: (qty, price). You set: {type(self.buy)}')
        if type(self.buy) is not np.ndarray:
            self.buy = self._get_formatted_order(self.buy)
        self._buy = self.buy.copy()

        if self.take_profit is not None:
            if store.exchanges.storage.get(self.exchange, None).type == 'spot':
                raise exceptions.InvalidStrategy(
                    "Setting self.take_profit in the go_long() method is not supported for spot trading (it's only supported in futures trading). "
                    "Try setting it in self.on_open_position() instead."
                )
                
            # validate
            self._validate_take_profit()

            self._prepare_take_profit()

        if self.stop_loss is not None:
            if store.exchanges.storage.get(self.exchange, None).type == 'spot':
                raise exceptions.InvalidStrategy(
                    "Setting self.stop_loss in the go_long() method is not supported for spot trading (it's only supported in futures trading). "
                    "Try setting it in self.on_open_position() instead."
                )
            # validate
            self._validate_stop_loss()

            self._prepare_stop_loss()

        # filters
        if not self._execute_filters():
            return

        self._submit_buy_orders()
        
    def _submit_buy_orders(self) -> None:
        cdef float price_to_compare = self.position.current_price
        # if jh.is_livetrading():
            # price_to_compare = jh.round_price_for_live_mode(
                # self.price,
                # selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision']
            # )
        # else:
            # price_to_compare = self.price
        for o in self._buy:
            # MARKET order
            # if fabsf(o[1] - price_to_compare) < 0.0001:
            if (fabsf(o[1] - price_to_compare)) < (fmaxf(0.0001*price_to_compare,0.001)):
                self.broker.buy_at_market(o[0])
            # STOP order
            elif o[1] > price_to_compare:
                self.broker.start_profit_at(sides.BUY, o[0], o[1])
            # LIMIT order
            elif o[1] < price_to_compare:
                self.broker.buy_at(o[0], o[1])
            else:
                raise ValueError(f'Invalid order price: o[1]:{o[1]}, self.price:{self.position.current_price}')

    def _submit_sell_orders(self) -> None:
        cdef float price_to_compare = self.position.current_price
        # if jh.is_livetrading():
            # price_to_compare = jh.round_price_for_live_mode(
                # self.price,
                # selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision']
            # )
        # else:
            # price_to_compare = self.price
        for o in self._sell:
            # MARKET order
            # if fabsf(o[1] - price_to_compare) < 0.0001:
            if (fabsf(o[1] - price_to_compare)) < (fmaxf(0.0001*price_to_compare,0.001)):
                self.broker.sell_at_market(o[0])
            # STOP order
            elif o[1] < price_to_compare:
                self.broker.start_profit_at(sides.SELL, o[0], o[1])
            # LIMIT order
            elif o[1] > price_to_compare:
                self.broker.sell_at(o[0], o[1])
            else:
                raise ValueError(f'Invalid order price: o[1]:{o[1]}, self.price:{self.position.current_price}')

    def _prepare_buy(self, make_copies: bool = True) -> None:
        try:
            self.buy = self._get_formatted_order(self.buy)
        except ValueError:
            raise exceptions.InvalidShape(
                'The format of self.buy is invalid. \n'
                f'It must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.buy} was given'
            )

        if make_copies:
            self._buy = self.buy.copy()

    def _prepare_sell(self, make_copies: bool = True) -> None:
        try:
            self.sell = self._get_formatted_order(self.sell)
        except ValueError:
            raise exceptions.InvalidShape(
                'The format of self.sell is invalid. \n'
                f'It must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.sell} was given'
            )

        if make_copies:
            self._sell = self.sell.copy()

    def _prepare_stop_loss(self, make_copies: bool = True) -> None:
        try:
            self.stop_loss = self._get_formatted_order(self.stop_loss)
        except ValueError:
            raise exceptions.InvalidShape(
                'The format of self.stop_loss is invalid. \n'
                f'It must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.stop_loss} was given'
            )

        if make_copies:
            self._stop_loss = self.stop_loss.copy()

    def _prepare_take_profit(self, make_copies: bool = True) -> None:
        try:
            self.take_profit = self._get_formatted_order(self.take_profit)
        except ValueError:
            raise exceptions.InvalidShape(
                'The format of self.take_profit is invalid. \n'
                f'It must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.take_profit} was given'
            )

        if make_copies:
            self._take_profit = self.take_profit.copy()
            
            
    def _validate_stop_loss(self) -> None:
        if self.stop_loss is None:
            raise exceptions.InvalidStrategy('You forgot to set self.stop_loss. example (qty, price)')
        elif type(self.stop_loss) not in [tuple, list, np.ndarray]:
            raise exceptions.InvalidStrategy(
                f'self.stop_loss must be either a list or a tuple. example: (qty, price). You set {type(self.stop_loss)}')

    def _validate_take_profit(self) -> None:
        if self.take_profit is None:
            raise exceptions.InvalidStrategy('You forgot to set self.take_profit. example (qty, price)')
        elif type(self.take_profit) not in [tuple, list, np.ndarray]:
            raise exceptions.InvalidStrategy(
                f'self.take_profit must be either a list or a tuple. example: (qty, price). You set {type(self.take_profit)}')

    def _execute_short(self) -> None:
        self.go_short()

        # validation
        if self.sell is None:
            raise exceptions.InvalidStrategy('You forgot to set self.sell. example (qty, price)')
        elif type(self.sell) not in [tuple, list]:
            raise exceptions.InvalidStrategy(
                f'self.sell must be either a list or a tuple. example: (qty, price). You set {type(self.sell)}'
            )

        if type(self.sell) is not np.ndarray:
            self.sell = self._get_formatted_order(self.sell)
        self._sell = self.sell.copy()

        if self.take_profit is not None:
            self._validate_take_profit()
            self._prepare_take_profit()

        if self.stop_loss is not None:
            self._validate_stop_loss()
            self._prepare_stop_loss()

        # filters
        if not self._execute_filters():
            return

        self._submit_sell_orders()

    def _execute_filters(self) -> bool:
        for f in self.filters():
            try:
                passed = f()
            except TypeError:
                raise exceptions.InvalidStrategy(
                    "Invalid filter format. You need to pass filter methods WITHOUT calling them "
                    "(no parentheses must be present at the end)"
                    "\n\n"
                    "\u274C " + "Incorrect Example:\n"
                                "return [\n"
                                "    self.filter_1()\n"
                                "]\n\n"
                                "\u2705 " + "Correct Example:\n"
                                            "return [\n"
                                            "    self.filter_1\n"
                                            "]\n"
                )

            if not passed:
                # logger.info(f.__name__)
                self._reset()
                return False

        return True

    @abstractmethod
    def go_long(self) -> None:
        pass

    # @abstractmethod
    def go_short(self) -> None:
        pass

    def _execute_cancel(self) -> None:
        """
        cancels everything so that the strategy can keep looking for new trades.
        """
        # validation
        if self.position.qty != 0 :
            raise Exception('cannot cancel orders when position is still open. there must be a bug somewhere.')

        # logger.info('cancel all remaining orders to prepare for a fresh start...')

        self.broker.cancel_all_orders()

        self._reset()

        self._broadcast('route-canceled')

        self.on_cancel()

        if not "pytest" in sys.modules:
            store.orders.storage[f'{self.exchange}-{self.symbol}'].clear()

    def _reset(self) -> None:
        self.buy = None
        self._buy = None
        self.sell = None
        self._sell = None
        self.stop_loss = None
        self._stop_loss = None
        self.take_profit = None
        self._take_profit = None

        self._executed_open_orders = []
        self._executed_close_orders = []
        store.orders.reset_trade_orders(self.exchange, self.symbol)
        
        self.increased_count = 0
        self.reduced_count = 0

    def before_start(self) -> None:
        """
        before the backtesting starts, update the strategy to prepare everything
        """
        pass
        
    def on_cancel(self) -> None:
        """
        what should happen after all active orders have been cancelled
        """
        pass

    @abstractmethod
    def should_long(self) -> bool:
        pass

    # @abstractmethod
    def should_short(self) -> bool:
        return False

    @abstractmethod
    def should_cancel_entry(self) -> bool:
        pass

    def before(self) -> None:
        """
        Get's executed BEFORE executing the strategy's logic
        """
        pass

    def after(self) -> None:
        """
        Get's executed AFTER executing the strategy's logic
        """
        pass

    def _update_position(self) -> None:
        # self._wait_until_executing_orders_are_fully_handled()

        # after _wait_until_executing_orders_are_fully_handled, the position might have closed, so:
        if self.position.qty == 0 :
            return
            
        self.update_position()

        self._detect_and_handle_entry_and_exit_modifications()

    def _detect_and_handle_entry_and_exit_modifications(self) -> None:
        cdef float sm_qty = self.position.qty 
        if sm_qty == 0:
            return
        try:
            if sm_qty > 0 and self.buy is not None:
                # prepare format
                if type(self.buy) is not np.ndarray:
                    self.buy = self._get_formatted_order(self.buy)

                # if entry has been modified
                if not arr_equal(self.buy,self._buy): #np.array_equal(self.buy, self._buy):
                    self._buy = self.buy.copy()

                    # cancel orders
                    for o in self.entry_orders:
                        if o.status == order_statuses.ACTIVE:
                            self.broker.cancel_order(o.id)
                    self._submit_buy_orders()

            elif sm_qty < 0 and self.sell is not None:
                # prepare format
                if type(self.sell) is not np.ndarray:
                    self.sell = self._get_formatted_order(self.sell)

                # if entry has been modified
                if not arr_equal(self.sell, self._sell): #np.array_equal(self.sell, self._sell):
                    self._sell = self.sell.copy()

                    # cancel orders
                    for o in self.entry_orders:
                        if o.status == order_statuses.ACTIVE:
                            self.broker.cancel_order(o.id)
                    self._submit_sell_orders()


            if sm_qty != 0 and self.take_profit is not None:
                self._validate_take_profit()
                self._prepare_take_profit(False)

                # if _take_profit has been modified
                if not arr_equal(self.take_profit, self._take_profit): #np.array_equal(self.take_profit, self._take_profit):
                    self._take_profit = self.take_profit.copy()

                    # if there's only one order in self._stop_loss, then it could be a liquidation order, store its price
                    if PyList_GET_SIZE(self._take_profit) == 1:
                        temp_current_price = self.price
                    else:
                        temp_current_price = None

                    # CANCEL previous orders
                    for o in self.exit_orders:
                        if o.is_take_profit and (o.status == order_statuses.ACTIVE):
                            self.broker.cancel_order(o.id)
                        elif o.is_executed:
                            self._executed_close_orders.append(o)
                    for o in self._take_profit:
                        if self.position.qty == 0:
                            break
                        # see if we need to override the take-profit price to be the current price to ensure a MARKET order
                        if temp_current_price == o[1]:
                            order_price = self.price
                        else:
                            order_price = o[1]

                        submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                        if submitted_order:
                            submitted_order.submitted_via = order_submitted_via.TAKE_PROFIT

            if sm_qty != 0 and self.stop_loss is not None:
                self._validate_stop_loss()
                self._prepare_stop_loss(False)

                # if stop_loss has been modified
                if not arr_equal(self.stop_loss, self._stop_loss): #np.array_equal(self.stop_loss, self._stop_loss):
                    # prepare format
                    self._stop_loss = self.stop_loss.copy()
                    
                    # if there's only one order in self._stop_loss, then it could be a liquidation order, store its price
                    if PyList_GET_SIZE(self._stop_loss) == 1:
                        temp_current_price = self.price
                    else:
                        temp_current_price = None
                        
                    # cancel orders
                    for o in self.exit_orders:
                        if o.is_stop_loss and (o.status == order_statuses.ACTIVE):
                            self.broker.cancel_order(o.id)
                    for o in self._stop_loss:
                        if self.position.qty == 0:
                            break
                        # see if we need to override the stop-loss price to be the current price to ensure a MARKET order
                        if temp_current_price == o[1]:
                            order_price = self.price
                        else:
                            order_price = o[1]

                        submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                        if submitted_order:
                            submitted_order.submitted_via = order_submitted_via.STOP_LOSS
        except TypeError and not "pytest" in sys.modules:
            raise exceptions.InvalidStrategy(
                'Something odd is going on within your strategy causing a TypeError exception. '
                'Try running it with the debug mode enabled in a backtest to see what was going on near the end, and fix it.'
            )
        except:
            raise

        # validations: stop-loss and take-profit should not be the same
        if (
                sm_qty != 0
                and (self.stop_loss is not None and self.take_profit is not None)
                and arr_equal(self.stop_loss, self.take_profit) #np.array_equal(self.stop_loss, self.take_profit)
        ):
            raise exceptions.InvalidStrategy(
                'stop-loss and take-profit should not be exactly the same. Just use either one of them and it will do.')

        
    def update_position(self) -> None:
        pass

    # def _wait_until_executing_orders_are_fully_handled(self):
        # if self._is_handling_updated_order:
            # logger.info(
                # "Stopped strategy execution at this time because we're still handling the result "
                # "of an executed order. Trying again in 3 seconds..."
            # )
            # sleep(0.1)

    def _check(self) -> None:
        """
        Based on the newly updated info, check if we should take action or not
        """
        if not self._is_initiated:
            self._is_initiated = True

        #self._wait_until_executing_orders_are_fully_handled()
            
        # should cancel entry?
        if self.should_cancel_entry() and PyList_GET_SIZE(self.entry_orders) > 0 and self.position.qty == 0:
            self._execute_cancel()

        # update position
        if self.position.qty != 0:
            # self._update_position()
            self.update_position()
            self._detect_and_handle_entry_and_exit_modifications()
            
        if config['app']['trading_mode'] == 'backtest' or "pytest" in sys.modules:
            if PyList_GET_SIZE(store.orders.to_execute) > 0:
                store.orders.execute_pending_market_orders()

        # should_long and should_short
        if self.position.qty == 0 and self.entry_orders == []:
            self._reset()
            should_short = self.should_short()
            # validate that should_short is not True if the exchange_type is spot
            if store.exchanges.storage.get(self.exchange, None).type == 'spot' and should_short is True:
                raise exceptions.InvalidStrategy(
                    'should_short cannot be True if the exchange type is "spot".'
                )
            should_long = self.should_long()
            # should_short and should_long cannot be True at the same time
            if should_short and should_long:
                raise exceptions.ConflictingRules(
                    'should_short and should_long should not be true at the same time.'
                )
            if should_long:
                self._execute_long()
            elif should_short:
                self._execute_short()
                
    @staticmethod
    def _simulate_market_order_execution() -> None:
        """
        Simulate market order execution in backtest mode
        """
        if jh.is_backtesting() or jh.is_unit_testing():
            if PyList_GET_SIZE(store.orders.to_execute) > 0:
                store.orders.execute_pending_market_orders()
            
    def _on_open_position(self, order: Order) -> None:
        self.increased_count = 1

        self._broadcast('route-open-position')

        if self.take_profit is not None:
            for o in self._take_profit:
                # validation: make sure take-profit will exit with profit, if not, close the position
                if self.position.qty > 0 and o[1] <= self.position.entry_price:
                    submitted_order: Order = self.broker.sell_at_market(o[0])
                    # logger.info(
                        # 'The take-profit is below entry-price for long position, so it will be replaced with a market order instead')
                elif self.position.qty < 0 and o[1] >= self.position.entry_price:
                    submitted_order: Order = self.broker.buy_at_market(o[0])
                    # logger.info(
                        # 'The take-profit is above entry-price for a short position, so it will be replaced with a market order instead')
                else:
                    submitted_order: Order = self.broker.reduce_position_at(o[0], o[1], self.position.current_price)
                if submitted_order:
                    submitted_order.submitted_via = order_submitted_via.TAKE_PROFIT

        if self.stop_loss is not None:
            for o in self._stop_loss:
                # validation: make sure stop-loss will exit with profit, if not, close the position
                if self.position.qty > 0 and o[1] >= self.position.entry_price:
                    submitted_order: Order = self.broker.sell_at_market(o[0])
                    # logger.info(
                        # 'The stop-loss is above entry-price for long position, so it will be replaced with a market order instead')
                elif self.position.qty < 0 and o[1] <= self.position.entry_price:
                    submitted_order: Order = self.broker.buy_at_market(o[0])
                    # logger.info(
                        # 'The stop-loss is below entry-price for a short position, so it will be replaced with a market order instead')
                else:
                    submitted_order: Order = self.broker.reduce_position_at(o[0], o[1], self.position.current_price)
                if submitted_order:
                    submitted_order.submitted_via = order_submitted_via.STOP_LOSS

        # self._entry_orders = []
        self.on_open_position(order)
        self._detect_and_handle_entry_and_exit_modifications()

    def on_open_position(self, order) -> None:
        """
        What should happen after the open position order has been executed
        """
        pass

    def on_close_position(self, order) -> None:
        """
        What should happen after the open position order has been executed
        """
        pass

    def _on_close_position(self, order: Order):

        self._broadcast('route-close-position')
        self._execute_cancel()
        self.on_close_position(order)

        self._detect_and_handle_entry_and_exit_modifications()

    def _on_increased_position(self, order: Order) -> None:
        self.increased_count += 1

        # self._entry_orders = []

        self._broadcast('route-increased-position')

        self.on_increased_position(order)

        self._detect_and_handle_entry_and_exit_modifications()

    def on_increased_position(self, order) -> None:
        """
        What should happen after the order (if any) increasing the
        size of the position is executed. Overwrite it if needed.
        And leave it be if your strategy doesn't require it
        """
        pass

    def _on_reduced_position(self, order: Order) -> None:
        """
        prepares for on_reduced_position() is implemented by user
        """
        self.reduced_count += 1

        # self._entry_orders = []

        self._broadcast('route-reduced-position')

        self.on_reduced_position(order)

        self._detect_and_handle_entry_and_exit_modifications()

    def on_reduced_position(self, order) -> None:
        """
        What should happen after the order (if any) reducing the size of the position is executed.
        """
        pass

    def on_route_open_position(self, strategy) -> None:
        """used when trading multiple routes that related

        Arguments:
            strategy {Strategy} -- the strategy that has fired (and not listening to) the event
        """
        pass

    def on_route_close_position(self, strategy) -> None:
        """used when trading multiple routes that related

        Arguments:
            strategy {Strategy} -- the strategy that has fired (and not listening to) the event
        """
        pass

    def on_route_increased_position(self, strategy) -> None:
        """used when trading multiple routes that related

        Arguments:
            strategy {Strategy} -- the strategy that has fired (and not listening to) the event
        """
        pass

    def on_route_reduced_position(self, strategy) -> None:
        """used when trading multiple routes that related

        Arguments:
            strategy {Strategy} -- the strategy that has fired (and not listening to) the event
        """
        pass

    def on_route_canceled(self, strategy) -> None:
        """used when trading multiple routes that related

        Arguments:
            strategy {Strategy} -- the strategy that has fired (and not listening to) the event
        """
        pass

    # @cython.wraparound(True)
    def _execute(self) -> None:
        """
        Handles the execution permission for the strategy.
        """
        cdef float sm_qty
        # make sure we don't execute this strategy more than once at the same time.
        if self._is_executing is True:
            return
                 
        self._is_executing = True

        self.before()
        
        # self._check()
        
        """
        Based on the newly updated info, check if we should take action or not
        """
        if not self._is_initiated:
            self._is_initiated = True

        #self._wait_until_executing_orders_are_fully_handled()
            
        # should cancel entry?
        if self.should_cancel_entry() and PyList_GET_SIZE(self.entry_orders) > 0 and self.position.qty == 0:
            self._execute_cancel()

        # update position
        if self.position.qty != 0:
            # self._update_position()
            self.update_position()
            # self._detect_and_handle_entry_and_exit_modifications()
            """ Detect and handle entry and exit modifications """
            sm_qty = self.position.qty 
            if sm_qty == 0:
                return
            try:
                if sm_qty > 0 and self.buy is not None:
                    # prepare format
                    if type(self.buy) is not np.ndarray:
                        self.buy = self._get_formatted_order(self.buy)

                    # if entry has been modified
                    if not arr_equal(self.buy,self._buy): #np.array_equal(self.buy, self._buy):
                        self._buy = self.buy.copy()

                        # cancel orders
                        for o in self.entry_orders:
                            if o.status == order_statuses.ACTIVE:
                                self.broker.cancel_order(o.id)
                        self._submit_buy_orders()

                elif sm_qty < 0 and self.sell is not None:
                    # prepare format
                    if type(self.sell) is not np.ndarray:
                        self.sell = self._get_formatted_order(self.sell)

                    # if entry has been modified
                    if not arr_equal(self.sell, self._sell): #np.array_equal(self.sell, self._sell):
                        self._sell = self.sell.copy()

                        # cancel orders
                        for o in self.entry_orders:
                            if o.status == order_statuses.ACTIVE:
                                self.broker.cancel_order(o.id)
                        self._submit_sell_orders()


                if sm_qty != 0 and self.take_profit is not None:
                    self._validate_take_profit()
                    self._prepare_take_profit(False)

                    # if _take_profit has been modified
                    if not arr_equal(self.take_profit, self._take_profit): #np.array_equal(self.take_profit, self._take_profit):
                        self._take_profit = self.take_profit.copy()

                        # if there's only one order in self._stop_loss, then it could be a liquidation order, store its price
                        if PyList_GET_SIZE(self._take_profit) == 1:
                            temp_current_price = self.price
                        else:
                            temp_current_price = None

                        # CANCEL previous orders
                        for o in self.exit_orders:
                            if o.is_take_profit and (o.status == order_statuses.ACTIVE):
                                self.broker.cancel_order(o.id)
                            elif o.is_executed:
                                self._executed_close_orders.append(o)
                        for o in self._take_profit:
                            if self.position.qty == 0:
                                break
                            # see if we need to override the take-profit price to be the current price to ensure a MARKET order
                            if temp_current_price == o[1]:
                                order_price = self.price
                            else:
                                order_price = o[1]

                            submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                            if submitted_order:
                                submitted_order.submitted_via = order_submitted_via.TAKE_PROFIT

                if sm_qty != 0 and self.stop_loss is not None:
                    self._validate_stop_loss()
                    self._prepare_stop_loss(False)

                    # if stop_loss has been modified
                    if not arr_equal(self.stop_loss, self._stop_loss): #np.array_equal(self.stop_loss, self._stop_loss):
                        # prepare format
                        self._stop_loss = self.stop_loss.copy()
                        
                        # if there's only one order in self._stop_loss, then it could be a liquidation order, store its price
                        if PyList_GET_SIZE(self._stop_loss) == 1:
                            temp_current_price = self.price
                        else:
                            temp_current_price = None
                            
                        # cancel orders
                        for o in self.exit_orders:
                            if o.is_stop_loss and (o.status == order_statuses.ACTIVE):
                                self.broker.cancel_order(o.id)
                        for o in self._stop_loss:
                            if self.position.qty == 0:
                                break
                            # see if we need to override the stop-loss price to be the current price to ensure a MARKET order
                            if temp_current_price == o[1]:
                                order_price = self.price
                            else:
                                order_price = o[1]

                            submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                            if submitted_order:
                                submitted_order.submitted_via = order_submitted_via.STOP_LOSS
            except TypeError and not "pytest" in sys.modules:
                raise exceptions.InvalidStrategy(
                    'Something odd is going on within your strategy causing a TypeError exception. '
                    'Try running it with the debug mode enabled in a backtest to see what was going on near the end, and fix it.'
                )
            except:
                raise

            # validations: stop-loss and take-profit should not be the same
            if (
                    sm_qty != 0
                    and (self.stop_loss is not None and self.take_profit is not None)
                    and arr_equal(self.stop_loss, self.take_profit) #np.array_equal(self.stop_loss, self.take_profit)
            ):
                raise exceptions.InvalidStrategy(
                    'stop-loss and take-profit should not be exactly the same. Just use either one of them and it will do.')

        """ End detect and handle entry and exit modifications """
            
            
        if config['app']['trading_mode'] == 'backtest' or "pytest" in sys.modules:
            if PyList_GET_SIZE(store.orders.to_execute) > 0:
                store.orders.execute_pending_market_orders()

        # should_long and should_short
        if self.position.qty == 0 and self.entry_orders == []:
            self._reset()
            should_short = self.should_short()
            # validate that should_short is not True if the exchange_type is spot
            if store.exchanges.storage.get(self.exchange, None).type == 'spot' and should_short is True:
                raise exceptions.InvalidStrategy(
                    'should_short cannot be True if the exchange type is "spot".'
                )
            should_long = self.should_long()
            # should_short and should_long cannot be True at the same time
            if should_short and should_long:
                raise exceptions.ConflictingRules(
                    'should_short and should_long should not be true at the same time.'
                )
            if should_long:
                self._execute_long()
            elif should_short:
                self._execute_short()
        """
        End Check
        """
                
                
        self.after()
        for m in self._cached_methods.values():
            m.cache_clear()

        self._is_executing = False
        self.index += 1

    def _terminate(self) -> None:
        """
        Optional for executing code after completion of a backTest.
        This block will not execute in live use as a live
        Jesse is never ending.
        """
        # if not (config['app']['trading_mode'] == 'optimize' or "pytest" in sys.modules): # or jh.is_debugging():
            # logger.info(f"Terminating {self.symbol}...")

        self.before_terminate()

        self._detect_and_handle_entry_and_exit_modifications()

        # fake execution of market orders in backtest simulation
        store.orders.execute_pending_market_orders()

        # if jh.is_live():
            # return

        if self.position.qty != 0:
            store.app.total_open_trades += 1
            store.app.total_open_pl += self.position.pnl
            # logger.info(
                # f"Closed open {self.exchange}-{self.symbol} position at {self.position.current_price} with PNL: {round(self.position.pnl, 4)}({round(self.position.pnl_percentage, 2)}%) because we reached the end of the backtest session."
            # )
            # first cancel all active orders so the balances would go back to the original state
            if store.exchanges.storage.get(self.exchange, None).type == 'spot':
                self.broker.cancel_all_orders()
            # fake a closing (market) order so that the calculations would be correct
            self.broker.reduce_position_at(self.position.qty, self.position.current_price, self.position.current_price)
            self.terminate()
            return

        if PyList_GET_SIZE(self.entry_orders) > 0:
            self._execute_cancel()
            # logger.info('Canceled open-position orders because we reached the end of the backtest session.')
            
        self.terminate()
        
    def before_terminate(self):
        pass
        
    def terminate(self):
        pass
        
    def watch_list(self) -> list:
        """
        returns an array containing an array of key-value items that should
        be logged when backTested, and monitored while liveTraded

        Returns:
            [array[{"key": v, "value": v}]] -- an array of dictionary objects
        """
        return []

    def _clear_cached_methods(self) -> None:
        for m in self._cached_methods.values():
            m.cache_clear()

    @property
    def current_candle(self) -> np.ndarray:
        """
        Returns current trading candle

        :return: np.ndarray
        """
        return store.candles.get_current_candle(self.exchange, self.symbol, self.timeframe).copy() 
        
    @property
    def open(self) -> float:
        """
        Returns the opening price of the current candle for this strategy.
        Just as a helper to use when writing super simple strategies.
        Returns:
            [float] -- the current trading candle's OPEN price
        """
        return self.current_candle[1]

    @property
    def close(self) -> float:
        """
        Returns the closing price of the current candle for this strategy.
        Just as a helper to use when writing super simple strategies.
        Returns:
            [float] -- the current trading candle's CLOSE price
        """
        return self.current_candle[2]

    @property
    def price(self) -> float:   
        """
        Same as self.close, except in livetrde, this is rounded as the exchanges require it.

        Returns:
            [float] -- the current trading candle's current(close) price
        """
        return self.position.current_price
        # return self.current_candle[2]

    @property
    def high(self) -> float:
        """
        Returns the highest price of the current candle for this strategy.
        Just as a helper to use when writing super simple strategies.
        Returns:
            [float] -- the current trading candle's HIGH price 
        """
        return self.current_candle[3]

    @property
    def low(self) -> float:
        """
        Returns the lowest price of the current candle for this strategy.
        Just as a helper to use when writing super simple strategies.
        Returns:
            [float] -- the current trading candle's LOW price
        """
        return self.current_candle[4]

    @property
    def candles(self) -> np.ndarray:
        """
        Returns candles for current trading route

        :return: np.ndarray
        """
        return store.candles.get_candles(self.exchange, self.symbol, self.timeframe) if not self.preload_candles else store.candles.storage[f'{self.exchange}-{self.symbol}-{self.timeframe}'].array[0:self.slice_amount[f'{self.exchange}-{self.symbol}-{self.timeframe}'] + self.index]

    def get_candles(self, exchange: str, symbol: str, timeframe: str) -> np.ndarray:
        """
        Get candles by passing exchange, symbol, and timeframe

        :param exchange: str
        :param symbol: str
        :param timeframe: str

        :return: np.ndarray
        """
        return store.candles.get_candles(exchange, symbol, timeframe) if not self.preload_candles else store.candles.storage[f'{self.exchange}-{self.symbol}-{self.timeframe}'].array[0:self.slice_amount[f'{self.exchange}-{self.symbol}-{self.timeframe}'] + self.index]

    @property
    def metrics(self) -> dict:
        """
        Returns all the metrics of the strategy.
        """
        if self.trades_count not in self._cached_metrics:
            self._cached_metrics[self.trades_count] = metrics.trades(
                store.completed_trades.trades, store.app.daily_balance, final=False
            )
        return self._cached_metrics[self.trades_count]

    @property
    def time(self) -> int:
        """returns the current time"""
        return store.app.time

    @property
    def balance(self) -> float:
        """the current capital in the trading exchange"""
        return self.position.exchange.wallet_balance

    @property
    def capital(self) -> float:
        """the current capital in the trading exchange"""
        raise NotImplementedError('The alias "self.capital" has been removed. Please use "self.balance" instead.')

    @property
    def available_margin(self) -> float:
        """Current available margin considering leverage"""
        return self.position.exchange.available_margin

    @property
    def fee_rate(self) -> float:
        return store.exchanges.storage.get(self.exchange, None).fee_rate  #selectors.get_exchange(self.exchange).fee_rate


    @property
    def is_long(self) -> bool:
        return self.position.qty > 0 #type == 'long'

    @property
    def is_short(self) -> bool:
        return self.position.qty < 0 #type == 'short'

    @property
    def is_open(self) -> bool:
        return self.position.qty != 0 

    @property
    def is_close(self) -> bool:
        return self.position.qty == 0 

    @property
    def average_stop_loss(self) -> float:
        if self._stop_loss is None:
            raise exceptions.InvalidStrategy('You cannot access self.average_stop_loss before setting self.stop_loss')

        arr = self._stop_loss
        return (np.abs(arr[:,0]* arr[:,1])).sum()/np.abs(arr[:,0]).sum()

    @property
    def average_take_profit(self) -> float:
        if self._take_profit is None:
            raise exceptions.InvalidStrategy(
                'You cannot access self.average_take_profit before setting self.take_profit')

        arr = self._take_profit
        return (np.abs(arr[:,0]*arr[:,1])).sum()/np.abs(arr[:,0]).sum()

    def _get_formatted_order(self, var) -> Union[list, np.ndarray]:

        if type(var) is np.ndarray:
            return var

        # just to make sure we also support None
        if var is None or var == []:
            return []

        # create a copy in the placeholders variables so we can detect future modifications
        # also, make it list of orders even if there's only one, to make it easier to loop
        if type(var[0]) not in [list, tuple]:
            var = [var]

        # create numpy array from list
        arr = np.array(var, dtype=float)

        # validate that the price (second column) is not less or equal to zero
        if arr[:, 1].min() <= 0:
            raise exceptions.InvalidStrategy(f'Order price must be greater than zero: \n{var}')

        return arr
        
    @property
    def average_entry_price(self) -> float:
        if self.position.qty > 0:
            arr = self._buy
        elif self.position.qty < 0:
            arr = self._sell
        elif self.has_long_entry_orders:
            arr = self._get_formatted_order(self.buy)
        elif self.has_short_entry_orders:
            arr = self._get_formatted_order(self.sell)
        else:
            return None
            
        # if type of arr is not np.ndarray, then it's not ready yet. Return None
        if type(arr) is not np.ndarray:
            arr = None

        if arr is None and self.position.qty != 0 :
            return self.position.entry_price
        elif arr is None:
            return None
            
        return (np.abs(arr[:,0]*arr[:,1])).sum()/np.abs(arr[:,0]).sum()

    @property
    def has_long_entry_orders(self) -> bool:
        # if no order has been submitted yet, but self.buy is not None, then we are calling
        # this property inside a filter.
        if self.entry_orders == [] and self.buy is not None:
            return True
        return self.entry_orders != [] and self.entry_orders[0].side == 'buy'

    @property
    def has_short_entry_orders(self) -> bool:
        # if no order has been submitted yet, but self.sell is not None, then we are calling
        # this property inside a filter.
        if self.entry_orders == [] and self.sell is not None:
            return True
        return self.entry_orders != [] and self.entry_orders[0].side == 'sell'
        
    @property    
    def average_open_price(self) -> float:
        executed = [[o.qty, o.price] for o in self._executed_open_orders] + [[o.qty, o.price] for o in self._entry_orders if o.is_executed]
        arr = self._convert_to_numpy_array(executed, 'self.average_open_price')
        if PyList_GET_SIZE(arr.shape) != 2:
            return None
        else:
            return (np.abs(arr[:, 0] * arr[:, 1])).sum() / np.abs(arr[:, 0]).sum()
    @property    
    def average_close_price(self) -> float:
        executed = [[o.qty, o.price] for o in self._executed_close_orders] + [[o.qty, o.price] for o in self._exit_orders if o.is_executed]
        arr = self._convert_to_numpy_array(executed, 'average_close_price')
        if PyList_GET_SIZE(arr.shape) != 2:
            return None
        else:
            return (np.abs(arr[:, 0] * arr[:, 1])).sum() / np.abs(arr[:, 0]).sum()

            
    def liquidate(self) -> None:
        """
        closes open position with a MARKET order
        """
        if self.position.qty == 0 :
            return

        if self.position.pnl > 0:
            self.take_profit = self.position.qty, self.position.current_price
        else:
            self.stop_loss = self.position.qty, self.position.current_price

    @property
    def shared_vars(self) -> dict:
        return store.vars

    @property
    def routes(self) -> List[Route]:
        return router.routes

    @property
    def leverage(self) -> int:
        if type(self.position.exchange) is SpotExchange:
            return 1
        elif type(self.position.exchange) is FuturesExchange:
            return self.position.exchange.futures_leverage
        else:
            raise ValueError('exchange type not supported!')

    @property
    def mark_price(self) -> float:
        return self.position.mark_price

    @property
    def funding_rate(self) -> float:
        return self.position.funding_rate

    @property
    def next_funding_timestamp(self) -> int:
        return self.position.next_funding_timestamp

    @property
    def liquidation_price(self) -> float:
        return self.position.liquidation_price

    @staticmethod
    def log(msg: str, log_type: str = 'info', send_notification: bool = False, webhook: str = None) -> None:
        msg = str(msg)

        if log_type == 'info':
            logger.info(msg, send_notification=jh.is_live() and send_notification, webhook=webhook)

        elif log_type == 'error':
            logger.error(msg, send_notification=jh.is_live() and send_notification)

        else:
            raise ValueError(f'log_type should be either "info" or "error". You passed {log_type}')
            
    @property
    def all_positions(self) -> Dict[str, Position]:
        positions_dict = {}
        for r in self.routes:
            positions_dict[r.symbol] = r.strategy.position
        return positions_dict

    @property
    def portfolio_value(self) -> float:
        total_position_values = 0
        exchange = store.exchanges.storage.get(self.exchange, None).type
        # in spot mode, self.balance does not include open order's value, so:

        # in spot mode, self.capital does not include open order's value, so:
        if exchange == 'spot':
            for o in self.entry_orders:
                if o.is_active:
                    total_position_values += o.value

            for key, p in self.all_positions.items():
                total_position_values += p.value

        # in futures mode, it's simpler:
        elif exchange == 'futures':
            for key, p in self.all_positions.items():
                total_position_values += p.pnl
        return (total_position_values + self.balance) * self.leverage
        

    @property
    def trades(self) -> List[ClosedTrade]:
        """
        Returns all the completed trades for this strategy.
        """
        return store.completed_trades.trades

    @property
    def orders(self) -> List[Order]:
        """
        Returns all the orders submitted by for this strategy.
        """
        key = f'{self.exchange}-{self.symbol}'
        all_orders = store.orders.storage.get(key, [])
        return all_orders #store.orders.get_orders(self.exchange, self.symbol)

    @property
    def entry_orders(self):
        """
        Returns all the entry orders for this position.
        """
        cdef list entry_orders, all_orders
        cdef Py_ssize_t all_orders_len, i 
        cdef str key = f'{self.exchange}-{self.symbol}'
        all_orders = store.orders.storage.get(key, [])
        all_orders_len = PyList_GET_SIZE(all_orders)
        if all_orders_len == 0:
            return PyList_New(0)
        p = store.positions.storage.get(key, None)
        cdef double c_qty = p.qty 
        if c_qty == 0 : 
            entry_orders = all_orders.copy()
        else:
            # entry_orders = [o for o in all_orders if o.side == jh.type_to_side(p.type)]    
            entry_orders = PyList_New(0)
            if c_qty > 0: 
                for i in range(all_orders_len):
                    if (all_orders[i].side == sides.BUY and not all_orders[i].status == order_statuses.CANCELED):
                        entry_orders.append(all_orders[i])
                    
                # entry_orders = [o for o in all_orders if (o.side == sides.BUY and not o.status == order_statuses.CANCELED)]
            elif c_qty < 0:
                for i in range(all_orders_len):
                    if (all_orders[i].side == sides.SELL and not all_orders[i].status == order_statuses.CANCELED):
                        entry_orders.append(all_orders[i])
                        
                        
                # entry_orders = [o for o in all_orders if (o.side == sides.SELL and not o.status == order_statuses.CANCELED)]
        
        # exclude cancelled orders
        # entry_orders = [o for o in entry_orders if not o.is_canceled]
        
        return entry_orders #store.orders.get_entry_orders(self.exchange, self.symbol)

    @property
    def exit_orders(self):
        """
        Returns all the exit orders for this position.
        """
        return store.orders.get_exit_orders(self.exchange, self.symbol)
        

    @property
    def exchange_type(self):   
        return store.exchanges.storage.get(self.exchange, None).type
        
    @property
    def is_spot_trading(self) -> bool:
        return store.exchanges.storage.get(self.exchange, None).type == 'spot'

    @property
    def is_futures_trading(self) -> bool:
        return store.exchanges.storage.get(self.exchange, None).type == 'futures'
        
    @property
    def daily_balances(self):
        return store.app.daily_balance
     
    @cython.wraparound(True)
    def check_precalculated_indicator_accuracy(self):
        round_num = 7
        if self._indicator10_value is not None:
            try:
                assert round(self._indicator10_value[-1],round_num) == round(self._indicator10_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value10: {round(self._indicator10_value[-1],round_num)} != { round(self._indicator10_test[-1],round_num)}')
        if self._indicator9_value is not None:
            try:
                assert round(self._indicator9_value[-1],round_num) == round(self._indicator9_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value9: {round(self._indicator9_value[-1],round_num)} != { round(self._indicator9_test[-1],round_num)}') 
        if self._indicator8_value is not None:
            try:
                assert round(self._indicator8_value[-1],round_num) == round(self._indicator8_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value8: {round(self._indicator8_value[-1],round_num)} != { round(self._indicator8_test[-1],round_num)}')   
        if self._indicator7_value is not None:        
            try:
                assert round(self._indicator7_value[-1],round_num) == round(self._indicator7_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value7: {round(self._indicator7_value[-1],round_num)} != { round(self._indicator7_test[-1],round_num)}')
        if self._indicator6_value is not None:
            try:
                assert round(self._indicator6_value[-1],round_num) == round(self._indicator6_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value6: {round(self._indicator6_value[-1],round_num)} != { round(self._indicator6_test[-1],round_num)}')
        if self._indicator5_value is not None:
            try:
                assert round(self._indicator5_value[-1],round_num) == round(self._indicator5_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value5: {round(self._indicator5_value[-1],round_num)} != { round(self._indicator5_test[-1],round_num)}')
        if self._indicator4_value is not None:
            try:
                assert round(self._indicator4_value[-1],round_num) == round(self._indicator4_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value4: {round(self._indicator4_value[-1],round_num)} != { round(self._indicator4_test[-1],round_num)}')
        if self._indicator3_value is not None:
            try:
                assert round(self._indicator3_value[-1],round_num) == round(self._indicator3_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value3: {round(self._indicator3_value[-1],round_num)} != { round(self._indicator3_test[-1],round_num)}')
        if self._indicator2_value is not None:
            try:
                assert round(self._indicator2_value[-1],round_num) == round(self._indicator2_test[-1],round_num)
            except AssertionError,error:
                print(f'stored value2: {round(self._indicator2_value[-1],round_num)} != { round(self._indicator2_test[-1],round_num)}')
        if self._indicator1_value is not None:
           try:
                assert round(self._indicator1_value[-1],round_num) == round(self._indicator1_test[-1],round_num)
           except AssertionError,error:
                print(f'stored value1: {round(self._indicator1_value[-1],round_num)} != { round(self._indicator1_test[-1],round_num)}')
            
            
    # def _gym_execute_before(self) -> None:
        # """
        # Split _execute method to 2. _gym_execute_before and _gym_execute
        # This way outside algorithm can inspect & inject such as a RL algorithm.
        # The inspection made using env_observation and injection will with agent_action.
        # """
        # if self._is_executing is True:
            # return

        # self._is_executing = True

        # self.before()

    # def _gym_execute(self) -> None:
        # """
        # Continue the _execute logic from _gym_execute_before method.
        # """
        # self._check()
        # self.after()
        # self._clear_cached_methods()

        # self._is_executing = False
        # self.index += 1

    # def observation_space(self) -> Space:
        # return spaces.Box(
            # low=-1, high=1,
            # shape=(1,),
            # dtype=np.int8
        # )

    # def env_observation(self):
        # """
        # get called in JessGymEnv class to get from the strategy what the user wants the RL will look like.
        # Make sure its like env_observation_space shape.
        # """
        # return []

    # def agent_action(self, action):
        # pass

    # def _agent_action(self, action: int):
        # """
        # translate the action
        # """
        # action = self.action_space()[action]
        # return self.agent_action(action)

    # def action_space(self) -> List[Enum]:
        # return [None]

    # def reward(self) -> float:
        # return 0.0