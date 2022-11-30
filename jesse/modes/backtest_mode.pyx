#cython:wraparound=True
#cython:boundscheck=False

import time
from typing import Dict, Union, List
# from guppy import hpy; h=hpy() 
import arrow
import numpy as np
cimport numpy as np 
import sys
from libc.math cimport fmin,fmax, NAN, abs, isnan, NAN
from numpy.math cimport INFINITY
import talib
cimport cython
np.import_array()
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t
ctypedef double dtype_t
ctypedef np.float64_t DTYPE_t
import pandas as pd
import random
def uuid4():
  s = '%032x' % random.getrandbits(128)
  return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
import jesse.helpers as jh
import jesse.services.metrics as stats
import jesse.services.required_candles as required_candles
# import jesse.services.selectors as selectors
# from guppy import hpy
from jesse import exceptions
from jesse.config import config
from jesse.enums import timeframes, order_types
from jesse.models import Candle, Order, Position
from jesse.modes.utils import save_daily_portfolio_balance
from jesse.routes import router
from jesse.services import charts
from jesse.services import quantstats
from jesse.services import report
from jesse.services.cache import cache
from jesse.services.candle import print_candle, candle_includes_price, split_candle
from jesse.services.candle import generate_candle_from_one_minutes
from jesse.services.numba_functions import monte_carlo_candles, stock_candles_func
from jesse.services.file import store_logs
from jesse.services.validators import validate_routes
from jesse.store import store
# from jesse.services import logger
from jesse.services.failure import register_custom_exception_handler
from jesse.services.redis import sync_publish, process_status
from timeloop import Timeloop
from datetime import timedelta
from jesse.services.progressbar import Progressbar
from jesse.enums import order_statuses

#get_fixed jump candle is disabled 

def run(
        debug_mode,
        user_config: dict,
        routes: List[Dict[str, str]],
        extra_routes: List[Dict[str, str]],
        start_date: str,
        finish_date: str,
        candles: dict = None,
        chart: bool = False,
        tradingview: bool = False,
        full_reports: bool = False,
        csv: bool = False,
        json: bool = False
) -> None:
    if not jh.is_unit_testing():
        # at every second, we check to see if it's time to execute stuff
        status_checker = Timeloop()
        @status_checker.job(interval=timedelta(seconds=1))
        def handle_time():
            if process_status() != 'started':
                raise exceptions.Termination
        status_checker.start()

    # import cProfile, pstats 
    # profiler = cProfile.Profile()
    # profiler.enable()    
    
    cdef list change,data
    cdef int routes_count, index
    # cdef float price_pct_change, bh_daily_returns_all_routes
    from jesse.config import config, set_config
    config['app']['trading_mode'] = 'backtest'

    # debug flag
    config['app']['debug_mode'] = debug_mode

    # inject config
    if not jh.is_unit_testing():
        set_config(user_config) 
    
    # set routes
    router.initiate(routes, extra_routes)
    
    store.app.set_session_id()

    register_custom_exception_handler()

    # validate routes
    validate_routes(router)

    # initiate candle store
    store.candles.init_storage(500000)

    # load historical candles
    if candles is None:
        candles = load_candles(start_date, finish_date)
        
    if not jh.should_execute_silently():
        sync_publish('general_info', {
            'session_id': jh.get_session_id(),
            'debug_mode': str(config['app']['debug_mode']),
        })

        # candles info
        key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
        sync_publish('candles_info', stats.candles_info(candles[key]['candles']))

        # routes info
        sync_publish('routes_info', stats.routes(router.routes))
        
    if router.routes[0].timeframe == '1m' or router.routes[0].timeframe == '2m':
        config['env']['simulation']['skip'] = False
        
    # run backtest simulation
    result = simulator(
        candles,
        run_silently=jh.should_execute_silently(),
        generate_charts=chart,
        generate_tradingview=tradingview,
        generate_quantstats=full_reports,
        generate_csv=csv,
        generate_json=json,
        generate_equity_curve=True,
        generate_hyperparameters=True,
        start_date = start_date,
        finish_date = finish_date
    )
    # print(result)

    if not jh.should_execute_silently():
        sync_publish('alert', {
            'message': f"Successfully executed backtest simulation in: {result['execution_duration']} seconds",
            'type': 'success'
        })
        sync_publish('hyperparameters', result['hyperparameters'])
        sync_publish('metrics', result['metrics'])
        sync_publish('equity_curve', result['equity_curve'])

    # profiler.disable()
    # pr_stats = pstats.Stats(profiler).sort_stats('tottime')
    # pr_stats.print_stats(50)
    # close database connection
    from jesse.services.db import database
    database.close_connection()

def _generate_quantstats_report(candles_dict: dict, finish_date: str, start_date: str) -> str:
    if store.completed_trades.count == 0:
        return None
    price_data = []
    timestamps = []
    # load close candles for Buy and hold and calculate pct_change
    for index, c in enumerate(config['app']['considering_candles']):
        exchange, symbol = c[0], c[1]
        if exchange in config['app']['trading_exchanges'] and symbol in config['app']['trading_symbols']:
            # fetch from database
            if exchange == ('Polygon_Stocks') :
                candles_tuple = stock_candles_func(symbol, start_date, finish_date)
            else:
                candles = candles_dict[jh.key(exchange, symbol)]['candles']

            if timestamps == []:
                timestamps = candles[:, 0]
            price_data.append(candles[:, 1])

    price_data = np.transpose(price_data)
    price_df = pd.DataFrame(
        price_data, index=pd.to_datetime(timestamps, unit="ms"), dtype=float
    ).resample('D').mean()
    price_pct_change = price_df.pct_change(1).fillna(0)
    buy_and_hold_daily_returns_all_routes = price_pct_change.mean(1)
    study_name = _get_study_name()
    res = quantstats.quantstats_tearsheet(buy_and_hold_daily_returns_all_routes, study_name)
    return res

def _get_study_name() -> str:
    routes_count = len(router.routes)
    more = f"-and-{routes_count - 1}-more" if routes_count > 1 else ""
    if type(router.routes[0].strategy_name) is str:
        strategy_name = router.routes[0].strategy_name
    else:
        strategy_name = router.routes[0].strategy_name.__name__
    study_name = f"{strategy_name}-{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}{more}"
    return study_name
    
@cython.wraparound(True)
def load_candles(start_date_str: str, finish_date_str: str) -> Dict[str, Dict[str, Union[str, np.ndarray]]]:
    cdef long start_date, finish_date, 
    cdef double required_candles_count
    cdef bint from_db
    cdef dict candles
    start_date = jh.date_to_timestamp(start_date_str)
    finish_date = jh.date_to_timestamp(finish_date_str) - 60000

    # validate
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError(
            "Can't load candle data from the future! The finish-date can be up to yesterday's date at most.")

    # load and add required warm-up candles for backtest
    if jh.is_backtesting():
        for c in config['app']['considering_candles']:
            exchange, symbol = c[0], c[1]
            exchange = required_candles.inject_required_candles_to_store(
                required_candles.load_required_candles(exchange, symbol, start_date_str, finish_date_str),
                exchange,
                symbol
            )

    # download candles for the duration of the backtest
    candles = {}
    for c in config['app']['considering_candles']:
        exchange, symbol = c[0], c[1]

        from_db = False
        key =  f'{exchange}-{symbol}'

        cache_key = f"{start_date_str}-{finish_date_str}-{key}"
        if jh.get_config('env.caching.recycle'):
            print('Recycling enabled!')
            cached_value = np.array(cache.slice_pickles(cache_key, start_date_str, finish_date_str, key))
        else:
            cached_value = np.array(cache.get_value(cache_key))
            print('Recycling disabled, falling back to vanilla driver!')
        if cached_value.any():
            candles_tuple = cached_value
        # if cache exists use cache_value
        # not cached, get and cache for later calls in the next 5 minutes
        # fetch from database
        else:
            if exchange == ('Polygon_Stocks') :
                print('stock candles being made')
                candles_tuple = stock_candles_func(symbol, start_date, finish_date)
            else: 
                print('candles being made')
                candles_tuple = Candle.select(
                        Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low,
                        Candle.volume
                    ).where(
                        Candle.exchange == exchange,
                        Candle.symbol == symbol,
                        Candle.timeframe == '1m' or Candle.timeframe.is_null(),
                        Candle.timestamp.between(start_date, finish_date)
                    ).order_by(Candle.timestamp.asc()).tuples()
            from_db = True
        # validate that there are enough candles for selected period
        required_candles_count = (finish_date - start_date) / 60_000
        if len(candles_tuple) == 0 or candles_tuple[-1][0] != finish_date or candles_tuple[0][0] != start_date:
            raise exceptions.CandleNotFoundInDatabase(
                f'Not enough candles for {symbol}. You need to import candles.'
            )
        elif len(candles_tuple) != required_candles_count + 1:
            raise exceptions.CandleNotFoundInDatabase(
                f'There are missing candles between {start_date_str} => {finish_date_str}')

        # cache it for near future calls
        if from_db:
            cache.set_value(cache_key, tuple(candles_tuple), expire_seconds=60 * 60 * 24 * 7)

        candles[key] = {
            'exchange': exchange,
            'symbol': symbol,
            'candles': np.array(candles_tuple)
        }

    return candles

def simulator(*args, **kwargs):	
    if jh.get_config('env.simulation.skip'):
        result = skip_simulator(*args, **kwargs)
    else:
        result = iterative_simulator(*args, **kwargs)	
    return result 

def iterative_simulator(
        candles: dict,
        run_silently: bool,
        hyperparameters: dict = None,
        generate_charts: bool = False,
        generate_tradingview: bool = False,
        generate_quantstats: bool = False,
        generate_csv: bool = False,
        generate_json: bool = False,
        generate_equity_curve: bool = False,
        generate_hyperparameters: bool = False,
        start_date: str = None,
        finish_date: str = None,
        full_path_name: str = None,
        full_name: str = None
) -> dict:
    result = {}
    cdef Py_ssize_t length, count
    cdef bint precalc_bool,indicator1_bool,indicator2_bool,indicator3_bool,indicator4_bool,indicator5_bool,indicator6_bool,indicator7_bool,indicator8_bool,indicator9_bool,indicator10_bool
    cdef np.ndarray indicator1_f, indicator2_f, indicator3_f, indicator4_f, indicator5_f, indicator6_f, indicator7_f, indicator8_f, indicator9_f,indicator10_f
    cdef dict indicator1_storage, indicator2_storage, indicator3_storage, indicator4_storage, indicator5_storage, indicator6_storage, indicator7_storage, indicator8_storage, indicator9_storage, indicator10_storage
    cdef int offset = 0
    cdef int total, f_offset
    begin_time_track = time.time()
    key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
    first_candles_set = candles[key]['candles']
    length = len(first_candles_set)
    # to preset the array size for performance
        
    try:
        store.app.starting_time = first_candles_set[0][0]
    except IndexError:
        raise IndexError('Check your "warm_up_candles" config value')
    store.app.time = first_candles_set[0][0]

    if jh.get_config('env.simulation.Montecarlo'):
        for j in candles:

            # candles[j]['candles'][:, 1], candles[j]['candles'][:, 2], candles[j]['candles'][:, 3], candles[j]['candles'][:, 4] = monte_carlo_candles(candles[j]['candles'][:])
            candles[j]['candles'][:, 1] = monte_carlo_candles(candles[j]['candles'][:, 1])
            candles[j]['candles'][:, 2] = monte_carlo_candles(candles[j]['candles'][:, 2])
            candles[j]['candles'][:, 3] = monte_carlo_candles(candles[j]['candles'][:, 3])
            candles[j]['candles'][:, 4] = monte_carlo_candles(candles[j]['candles'][:, 4])
            # candles[j]['candles'][:, 5] = monte_carlo_candles(candles[j]['candles'][:, 5])

    for r in router.routes:
        # if the r.strategy is str read it from file
        if isinstance(r.strategy_name, str):
            StrategyClass = jh.get_strategy_class(r.strategy_name)
        # else it is a class object so just use it
        else:
            StrategyClass = r.strategy_name

        try:
            r.strategy = StrategyClass()
        except TypeError:
            raise exceptions.InvalidStrategy(
                "Looks like the structure of your strategy directory is incorrect. Make sure to include the strategy INSIDE the __init__.py file."
                "\nIf you need working examples, check out: https://github.com/jesse-ai/example-strategies"
            )
        except:
            raise

        r.strategy.name = r.strategy_name
        r.strategy.exchange = r.exchange
        r.strategy.symbol = r.symbol
        r.strategy.timeframe = r.timeframe

        # read the dna from strategy's dna() and use it for injecting inject hyperparameters
        # first convert DNS string into hyperparameters
        if len(r.strategy.dna()) > 0 and hyperparameters is None:
            hyperparameters = jh.dna_to_hp(r.strategy.hyperparameters(), r.strategy.dna())

        # inject hyperparameters sent within the optimize mode
        if hyperparameters is not None:
            r.strategy.hp = hyperparameters

        # init few objects that couldn't be initiated in Strategy __init__
        # it also injects hyperparameters into self.hp in case the route does not uses any DNAs
        r.strategy._init_objects()
        key = f'{r.exchange}-{r.symbol}'
        store.positions.storage.get(key,None).strategy = r.strategy
        if full_path_name and full_name:
            r.strategy.full_path_name = full_path_name
            r.strategy.full_name = full_name

    # add initial balance
    save_daily_portfolio_balance()
    cdef Py_ssize_t i
    dic = {
        timeframes.MINUTE_1: 1,
        timeframes.MINUTE_2 : 2,
        timeframes.MINUTE_3: 3,
        timeframes.MINUTE_5: 5,
        timeframes.MINUTE_10 : 10,
        timeframes.MINUTE_15: 15,
        timeframes.MINUTE_30: 30,
        timeframes.MINUTE_45: 45,
        timeframes.HOUR_1: 60,
        timeframes.HOUR_2: 60 * 2,
        timeframes.HOUR_3: 60 * 3,
        timeframes.HOUR_4: 60 * 4,
        timeframes.HOUR_6: 60 * 6,
        timeframes.HOUR_8: 60 * 8,
        timeframes.HOUR_12: 60 * 12,
        timeframes.DAY_1: 60 * 24,
    }
    
    progressbar = Progressbar(length, step=60)
    indicator1_f = None
    indicator2_f = None
    indicator3_f = None
    indicator4_f = None
    indicator5_f = None
    indicator6_f = None
    indicator7_f = None
    indicator8_f = None
    indicator9_f = None
    indicator10_f = None 
    if jh.get_config('env.simulation.precalculation'):
        indicator1_storage,indicator2_storage,indicator3_storage,indicator4_storage,indicator5_storage,indicator6_storage,indicator7_storage,indicator8_storage,indicator9_storage,indicator10_storage = indicator_precalculation(candles,first_candles_set,store.positions.storage.get(key,None).strategy, False)
        indicator1_bool = True if indicator1_storage is not None else False
        indicator2_bool = True if indicator2_storage is not None else False
        indicator3_bool = True if indicator3_storage is not None else False
        indicator4_bool = True if indicator4_storage is not None else False
        indicator5_bool = True if indicator5_storage is not None else False
        indicator6_bool = True if indicator6_storage is not None else False
        indicator7_bool = True if indicator7_storage is not None else False
        indicator8_bool = True if indicator8_storage is not None else False
        indicator9_bool = True if indicator9_storage is not None else False
        indicator10_bool = True if indicator10_storage is not None else False
        precalc_bool = True
        f_offset = int(jh.get_config('env.simulation.previous_precalc_values'))
        if jh.get_config('env.simulation.precalc_test'):
            precalc_test = True
        else:
            precalc_test = False
    else:
        precalc_bool = False

    for i in range(length):
        # update time
        store.app.time = first_candles_set[i][0] + 60_000
        # add candles
        for j in candles:
            short_candle = candles[j]['candles'][i]
            # if i != 0:
                # previous_short_candle = candles[j]['candles'][i - 1]
                # if previous_short_candle[2] < short_candle[1]:
                    # short_candle[1] = previous_short_candle[2]
                    # short_candle[4] = fmin(previous_short_candle[2], short_candle[4])
                # elif previous_short_candle[2] > short_candle[1]:
                    # short_candle[1] = previous_short_candle[2]
                    # short_candle[3] = fmax(previous_short_candle[2], short_candle[3])
                # short_candle = short_candle
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']

            store.candles.add_one_candle(short_candle, exchange, symbol, '1m', with_execution=False,
                                     with_generation=False)

            # print short candle
            # if jh.is_debuggable('shorter_period_candles'):
                # print_candle(short_candle, True, symbol)
            _simulate_price_change_effect(short_candle, exchange, symbol)

            # generate and add candles for bigger timeframes
            for timeframe in config['app']['considering_timeframes']:
                # for 1m, no work is needed
                if timeframe == '1m':
                    continue

                count = dic[timeframe]
                # until = count - ((i + 1) % count)

                if (i + 1) % count == 0:
                    generated_candle = generate_candle_from_one_minutes(
                        candles[j]['candles'][(i - (count - 1)):(i + 1)])
                    store.candles.add_one_candle(generated_candle, exchange, symbol, timeframe, with_execution=False,
                                             with_generation=False)

        # update progressbar
        if not run_silently and i % 60 == 0:
            progressbar.update()
            sync_publish('progressbar', {
                'current': progressbar.current,
                'estimated_remaining_seconds': progressbar.estimated_remaining_seconds
            })

        # now that all new generated candles are ready, execute
        for r in router.routes:
            count = dic[r.timeframe]       
            indicator_key = f'{r.exchange}-{r.symbol}-{r.timeframe}'     
            # 1m timeframe
            if r.timeframe == timeframes.MINUTE_1:
                if precalc_bool:   
                    total = (i+1) 
                    if indicator1_bool:
                        indicator1_f = indicator1_storage[indicator_key][total-f_offset:total+1]
                    if indicator2_bool:
                        indicator2_f = indicator2_storage[indicator_key][total-f_offset:total+1]
                    if indicator3_bool:
                        indicator3_f = indicator3_storage[indicator_key][total-f_offset:total+1]
                    if indicator4_bool:
                        indicator4_f = indicator4_storage[indicator_key][total-f_offset:total+1]
                    if indicator5_bool:
                        indicator5_f = indicator5_storage[indicator_key][total-f_offset:total+1]
                    if indicator6_bool:
                        indicator6_f = indicator6_storage[indicator_key][total-f_offset:total+1]
                    if indicator7_bool:
                        indicator7_f = indicator7_storage[indicator_key][total-f_offset:total+1]
                    if indicator8_bool:
                        indicator8_f = indicator8_storage[indicator_key][total-f_offset:total+1]
                    if indicator9_bool:
                        indicator9_f = indicator9_storage[indicator_key][total-f_offset:total+1]
                    if indicator10_bool:
                        indicator10_f = indicator10_storage[indicator_key][total-f_offset:total+1]
                r.strategy._execute(indicator1_f,indicator2_f,indicator3_f,indicator4_f,indicator5_f,indicator6_f,indicator7_f,indicator8_f,indicator9_f,indicator10_f,precalc_bool,precalc_test)
            elif (i + 1) % count == 0:
                if precalc_bool: 
                    total = ((i/count)+1)
                    if indicator1_bool:
                        indicator1_f = indicator1_storage[indicator_key][total-f_offset:total+1]
                    if indicator2_bool:
                        indicator2_f = indicator2_storage[indicator_key][total-f_offset:total+1]
                    if indicator3_bool:
                        indicator3_f = indicator3_storage[indicator_key][total-f_offset:total+1]
                    if indicator4_bool:
                        indicator4_f = indicator4_storage[indicator_key][total-f_offset:total+1]
                    if indicator5_bool:
                        indicator5_f = indicator5_storage[indicator_key][total-f_offset:total+1]
                    if indicator6_bool:
                        indicator6_f = indicator6_storage[indicator_key][total-f_offset:total+1]
                    if indicator7_bool:
                        indicator7_f = indicator7_storage[indicator_key][total-f_offset:total+1]
                    if indicator8_bool:
                        indicator8_f = indicator8_storage[indicator_key][total-f_offset:total+1]
                    if indicator9_bool:
                        indicator9_f = indicator9_storage[indicator_key][total-f_offset:total+1]
                    if indicator10_bool:
                        indicator10_f = indicator10_storage[indicator_key][total-f_offset:total+1]
                r.strategy._execute(indicator1_f,indicator2_f,indicator3_f,indicator4_f,indicator5_f,indicator6_f,indicator7_f,indicator8_f,indicator9_f,indicator10_f,precalc_bool,precalc_test)
                
                # print candle
                # if jh.is_debuggable('trading_candles'):
                    # print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False,
                                 # r.symbol)
                                 
        # now check to see if there's any MARKET orders waiting to be executed
        store.orders.execute_pending_market_orders()

        if i != 0 and i % 1440 == 0: 
            save_daily_portfolio_balance()

    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        result['execution_duration'] = round(finish_time_track - begin_time_track, 2)

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest simulation is finished, add finishing balance
    save_daily_portfolio_balance()
    if generate_hyperparameters:
        result['hyperparameters'] = stats.hyperparameters(router.routes)
    result['metrics'] = report.portfolio_metrics()
    # generate logs in json, csv and tradingview's pine-editor format
    logs_path = store_logs(generate_json, generate_tradingview, generate_csv)
    if generate_json:
        result['json'] = logs_path['json']
    if generate_tradingview:
        result['tradingview'] = logs_path['tradingview']
    if generate_csv:
        result['csv'] = logs_path['csv']
    if generate_charts:
        result['charts'] = charts.portfolio_vs_asset_returns(_get_study_name())
    if generate_equity_curve:
        result['equity_curve'] = charts.equity_curve()
    if generate_quantstats:
        result['quantstats'] = _generate_quantstats_report(candles, start_date, finish_date)

    return result
    
# @cython.boundscheck(True)
cdef (double,double,double,double,double,double) c_sum(double [:,::1] array) nogil:  
    cdef Py_ssize_t i, rows
    cdef double sum1 = 0.0
    cdef double min1 = INFINITY
    cdef double max1 = -INFINITY
    cdef double close1, open1, time1
    close1 = array[-1,2] if array[-1,2] == array[-1,2] else NAN
    open1 = array[0,1] if array[0,1] == array[0,1] else NAN
    time1 = array[0,0]
    rows = array.shape[0]
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
        
    return sum1, min1, max1, close1, open1, time1

def generate_candles_from_minutes(double [:,::1] first_candles_set):
    sum1, min1, max1, close1, open1, time1 = c_sum(first_candles_set)
    return np.array([
        time1,
        open1,
        close1,
        max1,
        min1,
        sum1,
    ])
    
def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

    
def indicator_precalculation(dict candles,double [:,::1] first_candles_set,strategy, bint skip_1m):
    # import time 
    # start = time.time()
    
    cdef Py_ssize_t  i, consider_timeframes, candle_prestorage_shape, index, offset, length,rows, index2, candle_count
    cdef np.ndarray candle_prestorage, partial_array, gen_candles, partial_date_array, modified_partial_array
    cdef double [:,::1] new_candles, new_array, date_index
    cdef double [::1] indicator1_array, indicator2_array, indicator3_array, indicator4_array,indicator5_array,indicator6_array,indicator7_array,indicator8_array,indicator9_array,indicator10_array
    cdef bint stock_prices = False
    cdef bint trip 
    indicator1_storage = {}
    indicator2_storage = {}
    indicator3_storage = {}
    indicator4_storage = {}
    indicator5_storage = {}
    indicator6_storage = {}
    indicator7_storage = {}
    indicator8_storage = {}
    indicator9_storage = {}
    indicator10_storage = {}
    for j in candles:
        for timeframe in config['app']['considering_timeframes']:
            if (timeframe == '1m' and skip_1m): # or (len(config['app']['considering_timeframes']) > 1 and timeframe == '1m'):
                continue

            exchange = candles[j]['exchange']
            if exchange is {'Polygon_Stocks'}:
                stock_prices = True
            symbol = candles[j]['symbol']
            new_candles = candles[j]['candles']
            key = f'{exchange}-{symbol}-{timeframe}'
            consider_timeframes = jh.timeframe_to_one_minutes(timeframe)
            candle_prestorage = store.candles.storage[f'{exchange}-{symbol}-1m'].array
            candle_prestorage = trim_zeros(candle_prestorage) 
            candle_prestorage_shape = len(candle_prestorage)
            length = len(first_candles_set) + (candle_prestorage_shape)
            full_array = np.zeros((int(length/(consider_timeframes))+1,6))
            new_array = np.concatenate((candle_prestorage,new_candles),axis=0)
            partial_array = np.zeros((int(length/(consider_timeframes))+1,6))   
            partial_date_array = np.zeros((int(length/(consider_timeframes))+1,1))   
            index = 0
            index2 = 0
            candle_count = 0
            date_index = np.zeros([partial_array.shape[0],2]) 
            if stock_prices:
                for i in range(0,length):
                    if ((i + 1) % consider_timeframes == 0):
                        partial_array[(index)] = generate_candles_from_minutes(new_array[(i - (consider_timeframes-1)):(i+1)])
                        gen_candles = partial_array[(index)] 
                        if (gen_candles[5] == 0 and gen_candles[3] != 0 and gen_candles[2] != 0 and gen_candles[1] == gen_candles[2] and gen_candles[3] == gen_candles[2] and gen_candles[4] == gen_candles[2]):
                            index = index
                            index2 = index2 + 1
                        else: 
                            index = index + 1
                            index2 = index2
                        candle_count = candle_count + 1 
                        date_index[candle_count][0] = gen_candles[0]
                        date_index[candle_count][1] = index2
            else:
                for i in range(0,length):
                    if ((i + 1) % consider_timeframes == 0):
                        partial_array[(index)] = generate_candles_from_minutes(new_array[(i - (consider_timeframes-1)):(i+1)])
                        index = index + 1 
                        
            # print(index) 
            # print(date_index)
            # np.savetxt("test_indicator3.csv", partial_array, delimiter=",") 

            indicator1 = strategy._indicator1(precalc_candles = partial_array)
            indicator2 = strategy._indicator2(precalc_candles = partial_array)
            indicator3 = strategy._indicator3(precalc_candles = partial_array)
            indicator4 = strategy._indicator4(precalc_candles = partial_array)
            indicator5 = strategy._indicator5(precalc_candles = partial_array)
            indicator6 = strategy._indicator6(precalc_candles = partial_array)
            indicator7 = strategy._indicator7(precalc_candles = partial_array)
            indicator8 = strategy._indicator8(precalc_candles = partial_array)
            indicator9 = strategy._indicator9(precalc_candles = partial_array)
            indicator10 = strategy._indicator10(precalc_candles = partial_array)
            del new_candles
            del new_array
            del full_array
            if stock_prices:
                np.pad(date_index, (0, 3000), 'constant')
                for i in range(candle_count+((candle_prestorage_shape/consider_timeframes)-1)):
                    if i > candle_count:
                        i = candle_count
                    if date_index[i][1] != date_index[i-1][1]:   
                        indicator1 = np.insert(indicator1, i, indicator1[i-1]) 
                        indicator2 = np.insert(indicator2, i, indicator2[i-1]) 
                        indicator3 = np.insert(indicator3, i, indicator3[i-1]) 
                        indicator4 = np.insert(indicator4, i, indicator4[i-1]) 
                        indicator5 = np.insert(indicator5, i, indicator5[i-1]) 
                        indicator6 = np.insert(indicator6, i, indicator6[i-1]) 
                        indicator7 = np.insert(indicator7, i, indicator7[i-1]) 
                        indicator8 = np.insert(indicator8, i, indicator8[i-1]) 
                        indicator9 = np.insert(indicator9, i, indicator9[i-1]) 
                        indicator10 = np.insert(indicator10, i, indicator10[i-1]) 
                        
                            
            if indicator1 is not None:
                indicator1 = np.delete(indicator1,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator1_storage[key] = indicator1
            else:
                indicator1_storage = None
            if indicator2 is not None:
                indicator2 = np.delete(indicator2,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator2_storage[key] = indicator2  
            else:
                indicator2_storage = None
            if indicator3 is not None:
                indicator3 = np.delete(indicator3,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator3_storage[key] =  indicator3
            else:
                indicator3_storage = None
            if indicator4 is not None:
                indicator4 = np.delete(indicator4,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator4_storage[key] = indicator4
            else:
                indicator4_storage = None
            if indicator5  is not None:
                indicator5 = np.delete(indicator5,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator5_storage[key] = indicator5
            else:
                indicator5_storage = None
            if indicator6  is not None:
                indicator6 = np.delete(indicator6,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator6_storage[key] = indicator6 
            else:
                indicator6_storage = None
            if indicator7 is not None:
                indicator7 = np.delete(indicator7,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator7_storage[key] = indicator7            
            else:
                indicator7_storage = None
            if indicator8 is not None:
                indicator8 = np.delete(indicator8,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator8_storage[key] = indicator8 
            else:
                indicator8_storage = None
            if indicator9 is not None:
                indicator9 = np.delete(indicator9,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator9_storage[key] = indicator9 
            else:
                indicator9_storage = None
            if indicator10 is not None:
                indicator10 = np.delete(indicator10,slice(0,(candle_prestorage_shape/consider_timeframes)-1))
                indicator10_storage[key] = indicator10 
            else:
                indicator10_storage = None
    # end = time.time()
    # print(format(end-start))
    # print(len(partial_array))
    # print(len(indicator1))
    # print(indicator1.shape[0])
    # print(indicator1_storage)
    # print(f'candle_count: {candle_count}')
    # print(date_index.shape[0])
    # pd.DataFrame(date_index).to_csv('date_index.csv')
    # pd.DataFrame(partial_array).to_csv('candles.csv')
    # pd.DataFrame(indicator2).to_csv('indicator2.csv')
    return indicator1_storage, indicator2_storage, indicator3_storage, indicator4_storage, indicator5_storage, indicator6_storage, indicator7_storage, indicator8_storage, indicator9_storage, indicator10_storage

        
def skip_simulator(candles: dict,
        run_silently: bool,
        hyperparameters: dict = None,
        generate_charts: bool = False,
        generate_tradingview: bool = False,
        generate_quantstats: bool = False,
        generate_csv: bool = False,
        generate_json: bool = False,
        generate_equity_curve: bool = False,
        generate_hyperparameters: bool = False,
        start_date: str = None,
        finish_date: str = None,
        full_path_name: str= None,
        full_name: str = None
) -> dict:
    result = {}
    cdef Py_ssize_t i 
    cdef bint precalc_bool,indicator1_bool,indicator2_bool,indicator3_bool,indicator4_bool,indicator5_bool,indicator6_bool,indicator7_bool,indicator8_bool,indicator9_bool,indicator10_bool
    cdef dict indicator1_storage, indicator2_storage, indicator3_storage, indicator4_storage, indicator5_storage, indicator6_storage, indicator7_storage, indicator8_storage, indicator9_storage, indicator10_storage
    cdef int count, max_skip, length, min_timeframe_remainder, total, skip, generate_new_candle, f_offset
    cdef int offset = 0
    cdef np.ndarray indicator1_f, indicator2_f, indicator3_f, indicator4_f, indicator5_f, indicator6_f, indicator7_f, indicator8_f, indicator9_f,indicator10_f
    begin_time_track = time.time()
    key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
    first_candles_set = candles[key]['candles']
    length = len(first_candles_set)
    
    # to preset the array size for performance
    store.app.starting_time = first_candles_set[0][0]
    store.app.time = first_candles_set[0][0]
    # initiate strategies
    min_timeframe, strategy = _initialized_strategies(hyperparameters)
    if full_path_name and full_name:
        strategy.full_path_name = full_path_name
        strategy.full_name = full_name
    # add initial balance
    save_daily_portfolio_balance()
    
    i = min_timeframe_remainder = skip = min_timeframe
    cdef int update_dashboard = 240
    progressbar = Progressbar(length, step=min_timeframe * update_dashboard)
    # i is the i'th candle, which means that the first candle is i=1 etc..
    dic = {
        timeframes.MINUTE_1: 1,
        timeframes.MINUTE_2 : 2,
        timeframes.MINUTE_3: 3,
        timeframes.MINUTE_5: 5,
        timeframes.MINUTE_10 : 10,
        timeframes.MINUTE_15: 15,
        timeframes.MINUTE_30: 30,
        timeframes.MINUTE_45: 45,
        timeframes.HOUR_1: 60,
        timeframes.HOUR_2: 60 * 2,
        timeframes.HOUR_3: 60 * 3,
        timeframes.HOUR_4: 60 * 4,
        timeframes.HOUR_6: 60 * 6,
        timeframes.HOUR_8: 60 * 8,
        timeframes.HOUR_12: 60 * 12,
        timeframes.DAY_1: 60 * 24,
    }
    indicator1_f = None
    indicator2_f = None
    indicator3_f = None
    indicator4_f = None
    indicator5_f = None
    indicator6_f = None
    indicator7_f = None
    indicator8_f = None
    indicator9_f = None
    indicator10_f = None 
    if jh.get_config('env.simulation.precalculation'):
        indicator1_storage,indicator2_storage,indicator3_storage,indicator4_storage,indicator5_storage,indicator6_storage,indicator7_storage,indicator8_storage,indicator9_storage,indicator10_storage = indicator_precalculation(candles,first_candles_set,strategy,True)
        indicator1_bool = True if indicator1_storage is not None else False
        indicator2_bool = True if indicator2_storage is not None else False
        indicator3_bool = True if indicator3_storage is not None else False
        indicator4_bool = True if indicator4_storage is not None else False
        indicator5_bool = True if indicator5_storage is not None else False
        indicator6_bool = True if indicator6_storage is not None else False
        indicator7_bool = True if indicator7_storage is not None else False
        indicator8_bool = True if indicator8_storage is not None else False
        indicator9_bool = True if indicator9_storage is not None else False
        indicator10_bool = True if indicator10_storage is not None else False
        precalc_bool = True
        f_offset = int(jh.get_config('env.simulation.previous_precalc_values'))
        if jh.get_config('env.simulation.precalc_test'):
            precalc_test = True
        else:
            precalc_test = False
    else:
        precalc_bool = False

    
    while i <= length:
        # update time = open new candle, use i-1  because  0 < i <= length
        store.app.time = first_candles_set[i - 1][0] + 60_000

        # add candles
        for j in candles:
            short_candles = candles[j]['candles'][i - skip: i]
            # remove previous_short_candle fix
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']

            store.candles.add_multiple_candles(short_candles, exchange, symbol, '1m', with_execution=False,
                                     with_generation=False)

            # print short candle
            # if jh.is_debuggable('shorter_period_candles'):
                # print_candle(short_candles[-1], True, symbol)

            # only to check for a limit orders in this interval, its not necessary that the short_candles is the size of
            # any timeframe candle
            current_temp_candle = generate_candle_from_one_minutes(
                                                                   short_candles)

            # if i - skip > 0:
                # current_temp_candle = _get_fixed_jumped_candle(candles[j]['candles'][i - skip - 1],
                                                               # current_temp_candle)
                                                               
            # in this new prices update there might be an order that needs to be executed
            _simulate_price_change_effect(current_temp_candle, exchange, symbol)

            # generate and add candles for bigger timeframes
            for timeframe in config['app']['considering_timeframes']:
                # for 1m, no work is needed
                if timeframe == '1m':
                    continue

                # if timeframe is constructed by 1m candles without sync
                count = dic[timeframe]
                if i % count == 0:
                    generated_candle = generate_candle_from_one_minutes(
                        candles[j]['candles'][i - count:i])
                    store.candles.add_candle(generated_candle, exchange, symbol, timeframe, with_execution=False,
                                             with_generation=False)

        # update progressbar
        if not run_silently and i % (min_timeframe * update_dashboard) == 0:
            progressbar.update()
            sync_publish('progressbar', {
                'current': progressbar.current,
                'estimated_remaining_seconds': progressbar.estimated_remaining_seconds
            })

        # now that all new generated candles are ready, execute
        for r in router.routes:
            count = dic[r.timeframe]
            if i % count == 0:
                if precalc_bool:
                    total = (i/count)
                    indicator_key = f'{r.exchange}-{r.symbol}-{r.timeframe}' 
                    if indicator1_bool:
                        indicator1_f = indicator1_storage[indicator_key][total-f_offset:total+1]
                    if indicator2_bool:
                        indicator2_f = indicator2_storage[indicator_key][total-f_offset:total+1]
                    if indicator3_bool:
                        indicator3_f = indicator3_storage[indicator_key][total-f_offset:total+1]
                    if indicator4_bool:
                        indicator4_f = indicator4_storage[indicator_key][total-f_offset:total+1]
                    if indicator5_bool:
                        indicator5_f = indicator5_storage[indicator_key][total-f_offset:total+1]
                    if indicator6_bool:
                        indicator6_f = indicator6_storage[indicator_key][total-f_offset:total+1]
                    if indicator7_bool:
                        indicator7_f = indicator7_storage[indicator_key][total-f_offset:total+1]
                    if indicator8_bool:
                        indicator8_f = indicator8_storage[indicator_key][total-f_offset:total+1]
                    if indicator9_bool:
                        indicator9_f = indicator9_storage[indicator_key][total-f_offset:total+1]
                    if indicator10_bool:
                        indicator10_f = indicator10_storage[indicator_key][total-f_offset:total+1]
                # print candle
                # if jh.is_debuggable('trading_candles'):
                    # print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False,
                                 # r.symbol)
                r.strategy._execute(indicator1_f,indicator2_f,indicator3_f,indicator4_f,indicator5_f,indicator6_f,indicator7_f,indicator8_f,indicator9_f,indicator10_f,precalc_bool,precalc_test)
        # now check to see if there's any MARKET orders waiting to be executed
        store.orders.execute_pending_market_orders()

        if i % 1440 == 0:
            save_daily_portfolio_balance()

        skip = _skip_n_candles(candles, min_timeframe_remainder, i)
        if skip < min_timeframe_remainder:
            min_timeframe_remainder -= skip
        elif skip == min_timeframe_remainder:
            min_timeframe_remainder = min_timeframe
        i += skip

    res = 0
    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        result['execution_duration'] = round(finish_time_track - begin_time_track, 2)

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest simulation is finished, add finishing balance
    save_daily_portfolio_balance()
    
    if generate_hyperparameters:
        result['hyperparameters'] = stats.hyperparameters(router.routes)
    result['metrics'] = report.portfolio_metrics()
    # generate logs in json, csv and tradingview's pine-editor format
    logs_path = store_logs(generate_json, generate_tradingview, generate_csv)
    if generate_json:
        result['json'] = logs_path['json']
    if generate_tradingview:
        result['tradingview'] = logs_path['tradingview']
    if generate_csv:
        result['csv'] = logs_path['csv']
    if generate_charts:
        result['charts'] = charts.portfolio_vs_asset_returns(_get_study_name())
    if generate_equity_curve:
        result['equity_curve'] = charts.equity_curve()
    if generate_quantstats:
        result['quantstats'] = _generate_quantstats_report(candles, start_date, finish_date)

    return result
    
def _initialized_strategies(hyperparameters: dict = None):
    for r in router.routes:
        StrategyClass = jh.get_strategy_class(r.strategy_name)
        # print(StrategyClass)
        try:
            r.strategy = StrategyClass()
        except Exception as e:
            print(e)
            print(f'error {r.strategy_name}')
            raise exceptions.InvalidStrategy(
                "Looks like the structure of your strategy directory is incorrect. "
                "Make sure to include the strategy INSIDE the __init__.py file.\n"
                "If you need working examples, check out: https://github.com/jesse-ai/example-strategies"
            )

        r.strategy.name = r.strategy_name
        r.strategy.exchange = r.exchange
        r.strategy.symbol = r.symbol
        r.strategy.timeframe = r.timeframe
        # inject hyper parameters (used for optimize_mode)
        # convert DNS string into hyperparameters
        if len(r.strategy.dna()) > 0 and hyperparameters is None:
            hyperparameters = jh.dna_to_hp(r.strategy.hyperparameters(), r.strategy.dna())

        # inject hyperparameters sent within the optimize mode
        if hyperparameters is not None:
            r.strategy.hp = hyperparameters

        # init few objects that couldn't be initiated in Strategy __init__
        # it also injects hyperparameters into self.hp in case the route does not uses any DNAs
        r.strategy._init_objects()
        key = f'{r.exchange}-{r.symbol}'
        store.positions.storage.get(key,None).strategy = r.strategy

    # search for minimum timeframe for skips
    consider_timeframes = [jh.timeframe_to_one_minutes(timeframe) for timeframe in
                           config['app']['considering_timeframes'] if timeframe != '1m']
    # smaller timeframe is dividing DAY_1 & I down want bigger timeframe to be the skipper
    # because it fast enough with 1 day + higher timeframes are better to check every day ( 1M / 1W / 3D )
    if timeframes.DAY_1 not in consider_timeframes:
        consider_timeframes.append(jh.timeframe_to_one_minutes(timeframes.DAY_1))

    # for cases where only 1m is used in this simulation
    if not consider_timeframes:
        return 1
    # take the greatest common divisor for that purpose
    return np.gcd.reduce(consider_timeframes),r.strategy



cdef _finish_simulation(begin_time_track: float, run_silently: bool):
    res = 0
    if not run_silently:
        # print executed time for the backtest session
        finish_time_track = time.time()
        sync_publish('alert', {
            'message': f'Successfully executed backtest simulation in: {round(finish_time_track - begin_time_track, 2)} seconds',
            'type': 'success'
        })

    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()

    # now that backtest is finished, add finishing balance
    save_daily_portfolio_balance()

cdef int _skip_n_candles(candles, max_skip: int, i: int):
    """
    calculate how many 1 minute candles can be skipped by checking if the next candles
    will execute limit and stop orders
    Use binary search to find an interval that only 1 or 0 orders execution is needed
    :param candles: np.ndarray - array of the whole 1 minute candles
    :max_skip: int - the interval that not matter if there is an order to be updated or not.
    :i: int - the current candle that should be executed
    :return: int - the size of the candles in minutes needs to skip
    """
    cdef int orders_counter
    cdef list orders 
    while True:
        orders_counter = 0
        for r in router.routes:
            orders = store.orders.storage.get(f'{r.exchange}-{r.symbol}', [])
            if (sum(bool(o.status == order_statuses.ACTIVE) for o in orders)) < 2:
                continue

            # orders = store.orders.get_orders(r.exchange, r.symbol)
            future_candles = candles[f'{r.exchange}-{r.symbol}']['candles']
            if i >= len(future_candles):
                # if there is a problem with i or with the candles it will raise somewhere else
                # for now it still satisfy the condition that no more than 2 orders will be execute in the next candle
                break

            current_temp_candle = generate_candle_from_one_minutes(
                                                                   future_candles[i:i + max_skip])

            for order in orders:
                if order.status == order_statuses.ACTIVE and candle_includes_price(current_temp_candle, order.price):
                    orders_counter += 1

        if orders_counter < 2 or max_skip == 1:
            # no more than 2 orders that can interfere each other in this candle.
            # or the candle is 1 minute candle, so I cant reduce it to smaller interval :/
            break

        max_skip //= 2

    return max_skip
    
def _get_fixed_jumped_candle(previous_candle: np.ndarray, candle: np.ndarray) -> np.ndarray:
    """
    A little workaround for the times that the price has jumped and the opening
    price of the current candle is not equal to the previous candle's close!

    :param previous_candle: np.ndarray
    :param candle: np.ndarray
    """
    if previous_candle[2] < candle[1]:
        candle[1] = previous_candle[2]
        candle[4] = fmin(previous_candle[2], candle[4])
    elif previous_candle[2] > candle[1]:
        candle[1] = previous_candle[2]
        candle[3] = fmax(previous_candle[2], candle[3])

    return candle


def _simulate_price_change_effect(real_candle: np.ndarray, exchange: str, symbol: str) -> None:
    cdef bint executed_order
    cdef Py_ssize_t index
    # cdef str key 
    cdef np.ndarray current_temp_candle
    cdef list orders = store.orders.storage.get(f'{exchange}-{symbol}',[])
    cdef Py_ssize_t len_orders = len(orders)
    current_temp_candle = real_candle.copy()
    executed_order = False
    key = f'{exchange}-{symbol}'
    p = store.positions.storage.get(key, None)
    while True:
        if len_orders == 0:
            executed_order = False
        else:
            for index, order in enumerate(orders):
                if index == len_orders - 1 and not order.status == order_statuses.ACTIVE:
                    executed_order = False

                if not order.status == order_statuses.ACTIVE:
                    continue
                if (order.price >= current_temp_candle[4]) and (order.price <= current_temp_candle[3]): #candle_includes_price(current_temp_candle, order.price):
                    try:
                        storable_temp_candle, current_temp_candle = split_candle(current_temp_candle, order.price)
                    except Exception as e: 
                        print(e)
                        print(f'{current_temp_candle} - {order.price}')
                    store.candles.add_one_candle(
                        storable_temp_candle, exchange, symbol, '1m',
                        with_execution=False,
                        with_generation=False
                    )
                    # p = selectors.get_position(exchange, symbol)
                    p.current_price = storable_temp_candle[2]

                    executed_order = True

                    order.execute()

                    # break from the for loop, we'll try again inside the while
                    # loop with the new current_temp_candle
                    break
                else:
                    executed_order = False

        if not executed_order:
            # add/update the real_candle to the store so we can move on
            store.candles.add_one_candle(
                real_candle, exchange, symbol, '1m',
                with_execution=False,
                with_generation=False
            )
            # p = selectors.get_position(exchange, symbol)
            if p:
                p.current_price = real_candle[2]
            break
            
    p: Position = store.positions.storage.get(key, None)

    if not p:
        return

    # for now, we only support the isolated mode:
    if p.exchange.type == 'spot' or p.exchange.futures_leverage_mode == 'cross':
        return
        
    cdef double c_qty = p.qty
    cdef str c_type
    if c_qty == 0:
        c_liquidation_price = NAN
        c_type = 'close' 
    else:
        if c_qty > 0:   
            c_type = 'long'
            c_liquidation_price = p.entry_price * (1 - (1 / p.strategy.leverage) + 0.004)
        elif c_qty < 0:
            c_type = 'short'
            c_liquidation_price = p.entry_price * (1 + (1 / p.strategy.leverage) - 0.004)
        else:
            c_liquidation_price = NAN
            
    if (c_liquidation_price >= real_candle[4]) and (c_liquidation_price <= real_candle[3]):
        closing_order_side = jh.closing_side(c_type)

        # create the market order that is used as the liquidation order
        order = Order({
            'id':  uuid4(),
            'symbol': symbol,
            'exchange': exchange,
            'side': closing_order_side,
            'type': order_types.MARKET,
            'reduce_only': True,
            'qty': jh.prepare_qty(p.qty, closing_order_side),
            'price': p.bankruptcy_price
        })

        store.orders.add_order(order)

        store.app.total_liquidations += 1

        # logger.info(f'{p.symbol} liquidated at {p.liquidation_price}')

        order.execute()
            

    # _check_for_liquidations(real_candle, exchange, symbol)


def _check_for_liquidations(candle: np.ndarray, exchange: str, symbol: str) -> None:
    key = f'{exchange}-{symbol}'
    p: Position = store.positions.storage.get(key, None)

    if not p:
        return

    # for now, we only support the isolated mode:
    if p.mode != 'isolated':
        return

    if candle_includes_price(candle, p.liquidation_price):
        closing_order_side = jh.closing_side(p.type)

        # create the market order that is used as the liquidation order
        order = Order({
            'id':  uuid4(),
            'symbol': symbol,
            'exchange': exchange,
            'side': closing_order_side,
            'type': order_types.MARKET,
            'reduce_only': True,
            'qty': jh.prepare_qty(p.qty, closing_order_side),
            'price': p.bankruptcy_price
        })

        store.orders.add_order(order)

        store.app.total_liquidations += 1

        # logger.info(f'{p.symbol} liquidated at {p.liquidation_price}')

        order.execute()
