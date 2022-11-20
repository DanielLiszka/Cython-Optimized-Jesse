from typing import List, Dict
from jesse.services import charts 
import copy
from jesse.models import Candle
from jesse.modes.utils import save_daily_portfolio_balance
from jesse.routes import router
from jesse.services import quantstats
from jesse.services import report
import pandas as pd 
import numpy as np 
from jesse.services.file import store_logs

def backtest(
        config: dict,
        routes: List[Dict[str, str]],
        extra_routes: List[Dict[str, str]],
        candles: dict,
        generate_charts: bool = False,
        generate_tradingview: bool = False,
        generate_quantstats: bool = False,
        generate_hyperparameters: bool = False,
        generate_equity_curve: bool = False,
        generate_csv: bool = False,
        generate_json: bool = False,
        run_silently: bool = True,
        hyperparameters: dict = None,
        imported_string: str = None,
        optimizing: bool= None,
        full_path_name: str=None,
        full_name: str=None
) -> dict:

    # import cProfile, pstats 
    # profiler = cProfile.Profile()
    # profiler.enable()
    """
    An isolated backtest() function which is perfect for using in research, and AI training
    such as our own optimization mode. Because of it being a pure function, it can be used
    in Python's multiprocessing without worrying about pickling issues.

    Example `config`:
    {
        'starting_balance': 5_000,
        'fee': 0.001,
        'type': 'futures',
        'futures_leverage': 3,
        'futures_leverage_mode': 'cross',
        'exchange': 'Binance',
        'warm_up_candles': 100
    }

    Example `route`:
    [{'exchange': 'Bybit Perpetual', 'strategy': 'A1', 'symbol': 'BTC-USDT', 'timeframe': '1m'}]

    Example `extra_route`:
    [{'exchange': 'Bybit Perpetual', 'symbol': 'BTC-USDT', 'timeframe': '3m'}]

    Example `candles`:
    {
        'Binance-BTC-USDT': {
            'exchange': 'Binance',
            'symbol': 'BTC-USDT',
            'candles': np.array([]),
        },
    }
    """
    from jesse.services.validators import validate_routes
    from jesse.modes.backtest_mode import simulator #iterative_simulator as simulator 
    from jesse.config import config as jesse_config, reset_config
    from jesse.routes import router
    from jesse.store import store
    from jesse.store import install_routes
    from jesse.config import set_config
    from jesse.services import required_candles
    import jesse.helpers as jh
    
    jesse_config['app']['trading_mode'] = 'backtest'

    # inject (formatted) configuration values
    set_config(_format_config(config))

    # set routes
    router.initiate(routes, extra_routes)

    validate_routes(router)
    # TODO: further validate routes and allow only one exchange
    # TODO: validate the name of the exchange in the config and the route? or maybe to make sure it's a supported exchange

    # initiate candle store
    store.candles.init_storage(500000)
    
    # install_routes()
    
    # assert that the passed candles are 1m candles
    for key, value in candles.items():
        candle_set = value['candles']
        if candle_set[1][0] - candle_set[0][0] != 60_000:
            raise ValueError(
                f'Candles passed to the research.backtest() must be 1m candles. '
                f'\nIf you wish to trade other timeframes, notice that you need to pass it through '
                f'the timeframe option in your routes. '
                f'\nThe difference between your candles are {candle_set[1][0] - candle_set[0][0]} milliseconds which more than '
                f'the accepted 60000 milliseconds.'
            )
            
    cdef int warm_up_num
    # divide candles into warm_up_candles and trading_candles and then inject warm_up_candles
    max_timeframe = jh.max_timeframe(jesse_config['app']['considering_timeframes'])
    warm_up_num = config['warm_up_candles'] * jh.timeframe_to_one_minutes(max_timeframe)
    trading_candles = copy.deepcopy(candles)
    
    if warm_up_num != 0:
        for c in jesse_config['app']['considering_candles']:
            key = jh.key(c[0], c[1])
            # inject warm-up candles
            required_candles.inject_required_candles_to_store(
                candles[key]['candles'][:warm_up_num],
                c[0],
                c[1]
            )
            # update trading_candles
            trading_candles[key]['candles'] = candles[key]['candles'][warm_up_num:]
            
    #for testing purposes. this is used on 'run-silently' typically.
    start_date = '01-01-2020'
    finish_date = '01-06-2020'
    # run backtest simulation
    backtest_result = simulator(
        trading_candles,
        run_silently,
        hyperparameters=hyperparameters,
        generate_charts=generate_charts,
        generate_tradingview=generate_tradingview,
        generate_quantstats=generate_quantstats,
        generate_csv=generate_csv,
        generate_json=generate_json,
        generate_equity_curve=generate_equity_curve,
        generate_hyperparameters=generate_hyperparameters,
        start_date = start_date,
        finish_date = finish_date,
        full_path_name = full_path_name,
        full_name = full_name
    )

    result = {
        'metrics': {'total': 0, 'win_rate': 0, 'net_profit_percentage': 0},
        'charts': None,
        'logs': None,
    }
    if backtest_result['metrics'] is None:
        result['metrics'] = {'total': 0, 'win_rate': 0, 'net_profit_percentage': 0}
        result['logs'] = None
    else:
        result['metrics'] = backtest_result['metrics']
        result['logs'] = store.logs.info
        
    if generate_charts:
        result['charts'] = backtest_result['charts']
    if generate_tradingview:
        result['tradingview'] = backtest_result['tradingview']
    if generate_quantstats:
        result['quantstats'] = backtest_result['quantstats']
    if generate_csv:
        result['csv'] = backtest_result['csv']
    if generate_json:
        result['json'] = backtest_result['json']
    if generate_equity_curve:
        result['equity_curve'] = backtest_result['equity_curve']
    if generate_hyperparameters:
        result['hyperparameters'] = backtest_result['hyperparameters']
        
    if not run_silently:
        start_date = config['start_date']
        finish_date = config['finish_date']
        trial_num = config['trial_number']
        if store.completed_trades.count > 0:
            routes_count = len(router.routes)
            more = f"-and-{routes_count - 1}-more" if routes_count > 1 else ""
            study_name = f"{imported_string}-{router.routes[0].strategy_name}-{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}{more}-{start_date}-{finish_date}"
            store_logs(study_name)   
            full_reports = True
            path_name = f"{config['strategy_name']}-{config['exchange']}-{config['symbol']}-{config['timeframe']}-{config['start_date']}-{config['finish_date']}"
            # QuantStats' report
            if full_reports:
                price_data = []
                timestamps = []
                # load close candles for Buy and hold and calculate pct_change
                for index, c in enumerate(jesse_config['app']['considering_candles']):
                    exchange, symbol = c[0], c[1]
                    if exchange in jesse_config['app']['trading_exchanges'] and symbol in jesse_config['app']['trading_symbols']:
                        candles_ = trading_candles[jh.key(exchange, symbol)]['candles']
                        if timestamps == []:
                            timestamps = candles_[:, 0]
                        price_data.append(candles_[:, 1])

                price_data = np.transpose(price_data)
                price_df = pd.DataFrame(price_data, index=pd.to_datetime(timestamps, unit="ms"), dtype=float).resample(
                    'D').mean()
                price_pct_change = price_df.pct_change(1).fillna(0)
                bh_daily_returns_all_routes = price_pct_change.mean(1)
                quantstats.quantstats_tearsheet(bh_daily_returns_all_routes, study_name,optuna=True, path_name=path_name)

    # reset store and config so rerunning would be flawlessly possible
            from jesse.services.db import database
            database.close_connection()
            
    reset_config()
    store.reset()
    
    # profiler.disable()
    # pr_stats = pstats.Stats(profiler).sort_stats('tottime')
    # pr_stats.print_stats(50)
    if optimizing:
        return result['metrics']
    else:
        return result
    


def _format_config(config):
    """
    Jesse's required format for user_config is different from what this function accepts (so it
    would be easier to write for the researcher). Hence we need to reformat the config_dict:
    """
    exchange_config = {
        'balance': config['starting_balance'],
        'fee': config['fee'],
        'type': config['type'],
        'name': config['exchange'],
    }
    # futures exchange has different config, so:
    if exchange_config['type'] == 'futures':
        exchange_config['futures_leverage'] = config['futures_leverage']
        exchange_config['futures_leverage_mode'] = config['futures_leverage_mode']
        
    return {
        'exchanges': {
            config['exchange']: exchange_config
        },
        'logging': {
            'balance_update': True,
            'order_cancellation': True,
            'order_execution': True,
            'order_submission': True,
            'position_closed': True,
            'position_increased': True,
            'position_opened': True,
            'position_reduced': True,
            'shorter_period_candles': False,
            'trading_candles': True
        },
        'warm_up_candles': config['warm_up_candles']
    }
