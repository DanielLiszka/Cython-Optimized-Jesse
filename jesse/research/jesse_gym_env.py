import time
from copy import copy

import gym
import numpy as np
from gym import spaces

import jesse.helpers as jh

from typing import List, Dict

from jesse.store import store
from jesse.routes import router
from jesse.config import set_config, reset_config
from jesse.enums import timeframes
from jesse.config import config as jesse_config
from jesse.services.validators import validate_routes
from jesse.modes.utils import save_daily_portfolio_balance 
from jesse.services.candle import generate_candle_from_one_minutes 
from jesse.research.backtest import _format_config, _assert_candles, inject_warmup_and_extract_training_candles
from jesse.modes.backtest_mode import initialized_strategies, _simulate_price_change_effect, update_strategy_on_new_candle, _finish_simulation #,get_fixed_jumped_candle
from jesse.services.candle import candle_includes_price
# python3 'reinforced backtester script.py'

class JesseGymEnv(gym.Env):

    def __init__(self,
                 config: dict,
                 routes: List[Dict[str, str]],
                 extra_routes: List[Dict[str, str]],
                 candles: dict,
                 hyperparameters: dict = None,
                 generate_charts: bool = False,
                 generate_tradingview: bool = False,
                 generate_quantstats: bool = False,
                 generate_csv: bool = False,
                 generate_json: bool = False,
                 generate_equity_curve: bool = False,
                 generate_hyperparameters: bool = False,
                 generation_path: str = None,
                 episode_period_in_days=None,
                 ):
        """
        Initialization of the environment.
        Use the same arguments for the isolated_backtest from research module.
        """

        # assert that the passed candles are 1m candles
        # _assert_candles(candles)

        self.generate_csv = generate_csv
        self.generate_json = generate_json
        self.generate_charts = generate_charts
        self.generation_path = generation_path
        self.generate_quantstats = generate_quantstats
        self.generate_tradingview = generate_tradingview
        self.generate_equity_curve = generate_equity_curve
        self.generate_hyperparameters = generate_hyperparameters

        self.config = config
        self.routes = routes
        self.extra_routes = extra_routes
        self.hyperparameters = hyperparameters
        self.warm_n_training_candles = candles
        self.episode_period_in_days = episode_period_in_days

        self.i = None
        self.skip = None
        self.done = None 
        self.length = None
        self.candles = None
        self.strategy = None
        self.observation = None
        self.delay_base_i = None
        self.min_timeframe = None
        self.begin_time_track = None
        self.first_candles_set = None
        self.min_timeframe_remainder = None

        # reset() get called now because Strategy needs to be initialized for action and observation space
        self.action_space = None
        self.observation_space = None
        self.reset()

    def reset(self):
        self.begin_time_track = time.time()
        jesse_config['app']['trading_mode'] = 'backtest'

        # inject (formatted) configuration values
        set_config(_format_config(self.config))

        # check for timeframes
        for timeframe in jesse_config['app']['considering_timeframes']:
            if timeframe in [timeframes.DAY_3, timeframes.WEEK_1, timeframes.MONTH_1]:
                raise ValueError(f"{timeframe} timeframe not supported yet.")

        # set routes
        router.initiate(self.routes, self.extra_routes)

        if len(router.routes) != 1:
            raise IndexError('For now JesseGymEnv only support 1 route.')

        validate_routes(router)

        # initiate candle store
        store.candles.init_storage(5000)

        config_candles = copy(self.warm_n_training_candles)
        if self.episode_period_in_days is not None:
            # to avoid deepcopy of the numpy array :/ i do double copy
            candles_key = next(iter(config_candles.keys()))
            config_candles[candles_key] = copy(config_candles[candles_key])
            episode_candles = config_candles[candles_key]['candles']

            # look for the last day that the simulation can start with
            minutes_in_day = 1440
            last_possible_first_day = len(episode_candles) // minutes_in_day
            last_possible_first_day -= self.episode_period_in_days
            last_possible_first_day -= np.ceil(self.config['warm_up_candles']/minutes_in_day)
            random_start = np.random.randint(last_possible_first_day)
            random_start *= minutes_in_day

            episode_candles = episode_candles[random_start:random_start + self.episode_period_in_days*minutes_in_day]
            config_candles[candles_key]['candles'] = episode_candles

        # divide candles into warm_up_candles and trading_candles and then inject warm_up_candles
        self.candles = inject_warmup_and_extract_training_candles(
            self.config,
            config_candles,
            jesse_config['app']['considering_timeframes']
        )

        argmin_first_candle = np.inf
        self.first_candles_set = np.array([])
        for j in self.candles:
            if self.candles[j]['candles'][0, 0] < argmin_first_candle:
                self.first_candles_set = self.candles[j]['candles']
                argmin_first_candle = self.candles[j]['candles'][0, 0]
        self.length = len(self.first_candles_set)
        # to preset the array size for performance
        store.app.starting_time = self.first_candles_set[0, 0]
        store.app.time = self.first_candles_set[0, 0]

        # initiate strategies
        min_timeframe = initialized_strategies(self.hyperparameters)[0]
        self.strategy = router.routes[0].strategy

        self.action_space = spaces.Discrete(n=len(self.strategy.action_space()))
        self.observation_space = self.strategy.observation_space()

        # add initial balance
        save_daily_portfolio_balance()
        self.i = min_timeframe
        self.skip = min_timeframe
        self.min_timeframe = min_timeframe
        self.min_timeframe_remainder = min_timeframe

        # i' is the i'th candle, which means that the first candle is i=1 etc.
        # in case of more than 1 candle that starts in different times
        # for example DOT start from 1/1/21 and btc 10/12/20 so skip try to do something with dot candles
        self.delay_base_i = min(self.candles[j]['candles'][0, 0] for j in self.candles.keys())

        self.done = False

        self._step_new_candle()
        self.observation = self.strategy.env_observation()
        return self.observation

    def step(self, action):
        info = {}
        if self.done:
            raise ValueError('Environment should be reset! the simulation has already finished.')

        # inject action to the strategy!
        self.strategy._agent_action(action)

        # ======================
        # finish the candle from previous step using the action.
        # ======================
        self._step_finish_last_candle()

        if self.i > self.length:
            self.done = True
            self._close()
            return self.observation, self.strategy.reward(), self.done, info
 
        # ======================
        # start new candle here
        # ======================
        self._step_new_candle()

        self.observation = self.strategy.env_observation()

        return self.observation, self.strategy.reward(), False, info

    def render(self, mode="human"):
        pass

    def close(self):
        if not self.done:
            self.done = True
            self._close()

    def _close(self):
        backtest_result = _finish_simulation(self.begin_time_track,
                                            True,
                                            candles=self.candles,
                                            generate_charts=self.generate_charts,
                                            generate_tradingview=self.generate_tradingview,
                                            generate_quantstats=self.generate_quantstats,
                                            generate_csv=self.generate_csv,
                                            generate_json=self.generate_json,
                                            generate_equity_curve=self.generate_equity_curve,
                                            generate_hyperparameters=self.generate_hyperparameters,
                                            generation_path=self.generation_path
                                            )

        result = {
            'metrics': {'total': 0, 'win_rate': 0, 'net_profit_percentage': 0},
            'charts': None,
            'logs': None,
            'routes_metrics': backtest_result
        }

        if backtest_result['metrics'] is None:
            result['metrics'] = {'total': 0, 'win_rate': 0, 'net_profit_percentage': 0}
            result['logs'] = None
        else:
            result['metrics'] = backtest_result['metrics']
            result['logs'] = store.logs.info

        if self.generate_charts:
            result['charts'] = backtest_result['charts']
        if self.generate_tradingview:
            result['tradingview'] = backtest_result['tradingview']
        if self.generate_quantstats:
            result['quantstats'] = backtest_result['quantstats']
        if self.generate_csv:
            result['csv'] = backtest_result['csv']
        if self.generate_json:
            result['json'] = backtest_result['json']
        if self.generate_equity_curve:
            result['equity_curve'] = backtest_result['equity_curve']
        if self.generate_hyperparameters:
            result['hyperparameters'] = backtest_result['hyperparameters']

        # reset store and config so rerunning would be flawlessly possible
        reset_config()
        store.reset()

        return result

    def gym_execute_candles_before(self):
        r = router.routes[0]
        count = jh.timeframe_to_one_minutes(r.timeframe)
        if self.i % count == 0:
            self.strategy._gym_execute_before()

    def gym_execute_candles_continue(self):
        r = router.routes[0]
        count = jh.timeframe_to_one_minutes(r.timeframe)
        if self.i % count == 0:
            self.strategy._gym_execute()

        # now check to see if there's any MARKET orders waiting to be executed
        store.orders.execute_pending_market_orders()

    def _step_new_candle(self):
        # update time = open new candle, use i-1  because  0 < i <= length
        store.app.time = self.first_candles_set[self.i - 1][0] + 60_000

        # add candles
        for j in self.candles:
            delay_i = int((self.candles[j]['candles'][0, 0] - self.delay_base_i) // 60000)
            if self.i - self.skip < delay_i:
                # the date  of the simulation is not got to j yet
                continue
            short_candles = self.candles[j]['candles'][self.i - self.skip - delay_i: self.i - delay_i]
            # remove previous_short_candle fix
            exchange = self.candles[j]['exchange']
            symbol = self.candles[j]['symbol']

            store.candles.add_candle(short_candles, exchange, symbol, '1m', with_execution=False,
                                     with_generation=False)

            # only to check for a limit orders in this interval,
            # it is not necessary that the short_candles is the size of any timeframe candle
            current_temp_candle = generate_candle_from_one_minutes(
                                                                   short_candles)

            # if self.i - self.skip - delay_i > 0:
                # current_temp_candle = get_fixed_jumped_candle(
                    # self.candles[j]['candles'][self.i - self.skip - 1 - delay_i],
                    # current_temp_candle
                # )
            # in this new prices update there might be an order that needs to be executed
            _simulate_price_change_effect(current_temp_candle, exchange, symbol)

            # generate and add candles for bigger timeframes
            for timeframe in jesse_config['app']['considering_timeframes']:
                # for 1m, no work is needed
                if timeframe == '1m':
                    continue

                # if timeframe is constructed by 1m candles without sync
                count = jh.timeframe_to_one_minutes(timeframe)
                if self.i % count == 0:
                    candles_1m = store.candles.get_storage(exchange, symbol, '1m')
                    generated_candle = generate_candle_from_one_minutes(
                        candles_1m[len(candles_1m) - count:])
                    store.candles.add_candle(generated_candle, exchange, symbol, timeframe, with_execution=False,
                                             with_generation=False)
                    update_strategy_on_new_candle(generated_candle, exchange, symbol, timeframe)

        # now that all new generated candles are ready, execute
        self.gym_execute_candles_before()

    def _step_finish_last_candle(self):
        self.gym_execute_candles_continue()

        if self.i % 1440 == 0:
            save_daily_portfolio_balance()

        self.skip = self.skip_n_candles(self.candles, self.min_timeframe_remainder, self.i)
        if self.skip < self.min_timeframe_remainder:
            self.min_timeframe_remainder -= self.skip
        elif self.skip == self.min_timeframe_remainder:
            self.min_timeframe_remainder = self.min_timeframe
        self.i += self.skip

    def skip_n_candles(candles, max_skip: int, i: int) -> int:
        """
        calculate how many 1 minute candles can be skipped by checking if the next candles
        will execute limit and stop orders
        Use binary search to find an interval that only 1 or 0 orders execution is needed
        :param candles: np.ndarray - array of the whole 1 minute candles
        :max_skip: int - the interval that not matter if there is an order to be updated or not.
        :i: int - the current candle that should be executed
        :return: int - the size of the candles in minutes needs to skip
        """

        while True:
            orders_counter = 0
            for r in router.routes:
                if store.orders.count_active_orders(r.exchange, r.symbol) < 2:
                    continue

                orders = store.orders.get_orders(r.exchange, r.symbol)
                future_candles = candles[f'{r.exchange}-{r.symbol}']['candles']
                if i >= len(future_candles):
                    # if there is a problem with i or with the candles it will raise somewhere else
                    # for now it still satisfy the condition that no more than 2 orders will be execute in the next candle
                    break

                current_temp_candle = generate_candle_from_one_minutes('',
                                                                       future_candles[i:i + max_skip],
                                                                       accept_forming_candles=True)

                for order in orders:
                    if order.is_active and candle_includes_price(current_temp_candle, order.price):
                        orders_counter += 1

            if orders_counter < 2 or max_skip == 1:
                # no more than 2 orders that can interfere each other in this candle.
                # or the candle is 1 minute candle, so I cant reduce it to smaller interval :/
                break

            max_skip //= 2

        return max_skip
        