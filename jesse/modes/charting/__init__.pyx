import jesse.helpers as jh 
import numpy as np
cimport numpy as np 
from jesse.config import config
from typing import Dict, Union, List
cimport cython 
np.import_array() 
from jesse import exceptions 
from jesse.models import Candle 
from jesse.services.numba_functions import stock_candles_func
from jesse.services.cache import cache 
import arrow
from datetime import datetime, timedelta 
from jesse.enums import timeframes
from jesse.services.redis import sync_publish

timeframe_to_minutes = {
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
    
def charting(        
    user_config: dict,
    routes: Dict[str, str],
    start_date: str,
    finish_date: str,
    destination: list,
    indicator_info: dict = None
    ):
    
    candles = load_candles(start_date, finish_date, user_config,routes)
    start_date_formatted = datetime.strptime(start_date, "%Y-%m-%d")
    start_timestamp = int(start_date_formatted.timestamp())
    filtered_candles = candles[candles[:, 0] >= start_timestamp]
    charted_candlesticks = filtered_candles.tolist()
    
    # Return CandleSticks for now.
    print('returning data')
    #sync_publish('charting_candlesticks', charted_candlesticks, destination[0])
    return charted_candlesticks

    
    
@cython.wraparound(True)
def load_candles(start_date_str: str, finish_date_str: str, user_config: dict, routes:Dict[str, str]) -> np.ndarray:
    cdef long start_date, finish_date, 
    cdef double required_candles_count
    cdef bint from_db
    cdef dict candles
    
    date_obj = datetime.strptime(start_date_str, "%Y-%m-%d")
    timeframe = routes['timeframe']
    total_minutes = user_config['warm_up_candles'] * timeframe_to_minutes[timeframe]
    new_date_obj = date_obj - timedelta(minutes=total_minutes + 1440)
    start_date_str = new_date_obj.strftime("%Y-%m-%d")
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


    # download candles for the duration of the backtest
    candles = {}
    exchange = routes['exchange']
    symbol = routes['symbol']
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
        if exchange in ['Polygon_Stocks','Polygon_Forex'] :
            print('stock candles being made')
            candles_tuple = stock_candles_func(symbol, start_date, finish_date,exchange)
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

    return np.array(candles_tuple)