
import arrow
import numpy as np
from jesse.enums import timeframes
import jesse.helpers as jh
from jesse.config import config
from jesse.exceptions import CandleNotFoundInDatabase
from jesse.models import Candle
from jesse.services.cache import cache
from jesse.services.candle import generate_candle_from_one_minutes
from jesse.store import store
from jesse.services.numba_functions import stock_candles_func

def load_required_candles(exchange: str, symbol: str, start_date_str: str, finish_date_str: str) -> np.ndarray:
    cdef Py_ssize_t short_candles_count
    cdef int pre_finish_dates
    cdef long long start_date, finish_date, first_backtestable_timestamp
    """
    loads initial candles that are required before executing strategies.
    210 for the biggest timeframe and more for the rest
    """
    start_date = jh.arrow_to_timestamp(arrow.get(start_date_str, 'YYYY-MM-DD'))
    finish_date = jh.arrow_to_timestamp(arrow.get(finish_date_str, 'YYYY-MM-DD')) - 60000

    # validate
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError('Can\'t backtest the future!')

    max_timeframe = jh.max_timeframe(config['app']['considering_timeframes'])
    short_candles_count = jh.get_config('env.data.warmup_candles_num', 210) * jh.timeframe_to_one_minutes(max_timeframe)
    pre_finish_date = start_date - 60_000
    pre_start_date = pre_finish_date - short_candles_count * 60_000
    # make sure starting from the beginning of the day instead
    pre_start_date = jh.timestamp_to_arrow(pre_start_date).floor('day').int_timestamp * 1000
    # update candles_count to count from the beginning of the day instead
    short_candles_count = int((pre_finish_date - pre_start_date) / 60_000)

    key = jh.key(exchange, symbol)
    cache_key = f'{jh.timestamp_to_date(pre_start_date)}-{jh.timestamp_to_date(pre_finish_date)}-{key}'
    cached_value = np.array(cache.get_value(cache_key))
    # if cache exists
    if cached_value.any():
        candles_tuple = cached_value
    # not cached, get and cache for later calls in the next 5 minutes
    else:
        # fetch from database
        if exchange == ('Polygon_Stocks') :
            candles_tuple = stock_candles_func(symbol, pre_start_date, pre_finish_date)
        else:
            candles_tuple = tuple(
                Candle.select(
                    Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low,
                    Candle.volume
                ).where(
                    Candle.exchange == exchange,
                    Candle.symbol == symbol,
                    Candle.timeframe == '1m' or Candle.timeframe.is_null(),
                    Candle.timestamp.between(pre_start_date, pre_finish_date)
                ).order_by(Candle.timestamp.asc()).tuples()
            )

        # cache it for near future calls
        cache.set_value(cache_key, candles_tuple, expire_seconds=60 * 60 * 24 * 7)

    candles = np.array(candles_tuple)
    if len(candles) < short_candles_count + 1:
        first_existing_candle = tuple(
            Candle.select(Candle.timestamp).where(
                Candle.exchange == exchange,
                Candle.symbol == symbol,
                Candle.timeframe == '1m' or Candle.timeframe.is_null()
            ).order_by(Candle.timestamp.asc()).limit(1).tuples()
        )

        if not len(first_existing_candle):
            raise CandleNotFoundInDatabase(
                f'No candle for {exchange} {symbol} is present in the database. Try importing candles.'
            )

        first_existing_candle = first_existing_candle[0][0]

        last_existing_candle = tuple(
            Candle.select(Candle.timestamp).where(
                Candle.exchange == exchange,
                Candle.symbol == symbol,
                Candle.timeframe == '1m' or Candle.timeframe.is_null()
            ).order_by(Candle.timestamp.desc()).limit(1).tuples()
        )[0][0]

        first_backtestable_timestamp = first_existing_candle + (pre_finish_date - pre_start_date) + (60_000 * 1440)

        # if first backtestable timestamp is in the future, that means we have some but not enough candles
        if first_backtestable_timestamp > jh.today_to_timestamp():
            raise CandleNotFoundInDatabase(
                f'Not enough candle for {exchange} {symbol} is present in the database. Jesse requires "210 * biggest_timeframe" warm-up candles. '
                'Try importing more candles from an earlier date.'
            )

        raise CandleNotFoundInDatabase(
            f'Not enough candles for {exchange} {symbol} exists to run backtest from {start_date_str} => {finish_date_str}. \n'
            f'First available date is {jh.timestamp_to_date(first_backtestable_timestamp)}\n'
            f'Last available date is {jh.timestamp_to_date(last_existing_candle)}'
        )

    return candles


def inject_required_candles_to_store(candles: np.ndarray, exchange: str, symbol: str) -> None:
    # cdef int candle_counter
    """
    generate and add required candles to the candle store
    """
    # batch add 1m candles:
    store.candles.batch_add_candle(candles, exchange, symbol, '1m', with_generation=False)
    # loop to generate, and add candles (without execution)
    # candle_counter = -1
    cdef dict dic = {
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

    for i in range(len(candles)):
        for timeframe in config['app']['considering_timeframes']:
            # skip 1m. already added
            if timeframe == '1m':
                continue

            num = dic[timeframe]

            if (i + 1) % num == 0:
                # candle_counter = candle_counter + 1 
                generated_candle = generate_candle_from_one_minutes(
                    candles[(i - (num - 1)):(i + 1)],
                )

                store.candles.add_candle(
                    generated_candle,
                    exchange,
                    symbol,
                    timeframe,
                    with_execution=False,
                    with_generation=False
                )
    return exchange