#cython: boundscheck= True
#cython: wraparound= True
#cython : nonetype =  True
#cython : cdivision = False

from datetime import datetime, timedelta
from typing import List, Any, Union 
from jesse.models import ClosedTrade
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport NAN, abs, isnan, fmax
import pandas as pd
from quantstats import stats

import jesse.helpers as jh
from jesse.store import store
from jesse.services import selectors

def candles_info(candles_array: np.ndarray) -> dict:
    cdef unsigned long period = jh.date_diff_in_days(jh.timestamp_to_arrow(candles_array[0][0]),jh.timestamp_to_arrow(candles_array[-1][0])) + 1

    if period > 365:
        duration = f'{period} days ({round(period / 365, 2)} years)'
    elif period > 30:
        duration = f'{period} days ({round(period / 30, 2)} months)'
    else:
        duration = f'{period} days'

   # type of the exchange
    trading_exchange = selectors.get_trading_exchange()

    info = {
        'duration': duration,
        'starting_time': candles_array[0][0],
        'finishing_time': (candles_array[-1][0] + 60_000),
        'exchange_type': trading_exchange.type,
    }

    # if the exchange type is futures, also display leverage
    if trading_exchange.type == 'futures':
        info['leverage'] = trading_exchange.futures_leverage
        info['leverage_mode'] = trading_exchange.futures_leverage_mode

    return info


def routes(routes_arr: list) -> list:
    return [{
            'exchange': r.exchange,
            'symbol': r.symbol,
            'timeframe': r.timeframe,
            'strategy_name': r.strategy_name,
        } for r in routes_arr]


def trades(trades_list: List[ClosedTrade], daily_balance: list, final: bool = True) -> dict:

    cdef double starting_balance, current_balance, s_max, s_min,fee,net_profit,net_profit_percentage,average_win,average_loss,profit_factor,expectancy,expectancy_percentage,average_holding_period,average_winning_holding_period,average_losing_holding_period,gross_profit,gross_loss,open_pl
    cdef unsigned long total_completed, total_winning_trades, total_losing_trades, longs_count, shorts_count, total_open_trades
    
    starting_balance = 0
    current_balance = 0

    for e in store.exchanges.storage:
        starting_balance += store.exchanges.storage[e].starting_assets[jh.app_currency()]
        current_balance += store.exchanges.storage[e].assets[jh.app_currency()]

    if not trades_list:
        return {'total': 0, 'win_rate': 0, 'net_profit_percentage': 0}

    df = pd.DataFrame.from_records([t.to_dict for t in trades_list])
    total_completed = len(df)
    winning_trades = df.loc[df['PNL'] > 0]
    total_winning_trades = len(winning_trades)
    losing_trades = df.loc[df['PNL'] < 0]
    total_losing_trades = len(losing_trades)

    arr = df['PNL'].to_numpy()
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    current_streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                              -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

    s_min = current_streak.min()
    losing_streak = 0 if s_min > 0 else abs(s_min)

    s_max = current_streak.max()
    winning_streak = fmax(s_max, 0)

    largest_losing_trade = 0 if total_losing_trades == 0 else losing_trades['PNL'].min()
    largest_winning_trade = 0 if total_winning_trades == 0 else winning_trades['PNL'].max()

    if len(winning_trades) == 0:
        win_rate = 0
    else:
        win_rate = len(winning_trades) / (len(losing_trades) + len(winning_trades))
    longs_count = len(df.loc[df['type'] == 'long'])
    shorts_count = len(df.loc[df['type'] == 'short'])
    longs_percentage = longs_count / (longs_count + shorts_count) * 100
    shorts_percentage = 100 - longs_percentage
    fee = df['fee'].sum()
    net_profit = df['PNL'].sum()
    net_profit_percentage = (net_profit / starting_balance) * 100
    avg_win_percentage = df.loc[df['PNL'] > 0, 'PNL_percentage'].sum() / len(winning_trades)
    avg_loss_percentage = -df.loc[df['PNL'] < 0, 'PNL_percentage'].sum() / len(losing_trades)
    average_win = winning_trades['PNL'].mean()
    average_loss = abs(losing_trades['PNL'].mean())
    ratio_avg_win_loss = average_win / average_loss
    expectancy = (0 if isnan(average_win) else average_win) * win_rate - (
        0 if isnan(average_loss) else average_loss) * (1 - win_rate)
    expectancy = expectancy
    expectancy_percentage = (expectancy / starting_balance) * 100
    expected_net_profit_every_100_trades = expectancy_percentage * 100
    average_holding_period = df['holding_period'].mean()
    average_winning_holding_period = winning_trades['holding_period'].mean()
    average_losing_holding_period = losing_trades['holding_period'].mean()
    gross_profit = winning_trades['PNL'].sum()
    gross_loss = losing_trades['PNL'].sum()
    profit_factor = 0 if gross_loss == 0 else (gross_profit/(abs(gross_loss)))
    kelly_criterion = win_rate - ((1 - win_rate) / ratio_avg_win_loss)
    
    start_date = pd.Timestamp.fromtimestamp(store.app.starting_time /1000) #datetime.fromtimestamp(store.app.starting_time / 1000)
    date_index = pd.date_range(start=start_date, periods=len(daily_balance))

    daily_return = pd.DataFrame(daily_balance, index=date_index).pct_change(1)

    total_open_trades = store.app.total_open_trades
    open_pl = store.app.total_open_pl
    

    max_drawdown = NAN
    annual_return = NAN
    sharpe_ratio = NAN
    calmar_ratio = NAN
    sortino_ratio = NAN
    omega_ratio = NAN
    serenity_index = NAN
    smart_sharpe = NAN
    smart_sortino = NAN

    if len(daily_return) > 2:
        max_drawdown = stats.max_drawdown(daily_return).values[0] * 100
        annual_return = stats.cagr(daily_return).values[0] * 100
        sharpe_ratio = stats.sharpe(daily_return, periods=365).values[0]
        calmar_ratio = stats.calmar(daily_return).values[0]
        sortino_ratio = stats.sortino(daily_return, periods=365).values[0]
        omega_ratio = stats.omega(daily_return, periods=365)
        serenity_index = stats.serenity_index(daily_return).values[0]
        # As those calculations are slow they are only done for the final report and not at self.metrics in the strategy.
        if final:
            smart_sharpe = stats.smart_sharpe(daily_return, periods=365).values[0]
            smart_sortino = stats.smart_sortino(daily_return, periods=365).values[0]

    return {
        'total': NAN if (total_completed != total_completed) else total_completed,
        'total_winning_trades': NAN if (total_winning_trades != total_winning_trades) else total_winning_trades,
        'total_losing_trades': NAN if (total_losing_trades != total_losing_trades) else total_losing_trades,
        'starting_balance': NAN if (starting_balance != starting_balance) else starting_balance,
        'finishing_balance': NAN if (current_balance != current_balance) else current_balance,
        'win_rate': NAN if (win_rate != win_rate) else win_rate,
        # 'avg_win_percentage': NAN if (avg_win_percentage != avg_win_percentage) else avg_win_percentage,
        # 'avg_loss_percentage': NAN if (avg_loss_percentage != avg_loss_percentage) else avg_loss_percentage,
        'profit_factor' : NAN if (profit_factor != profit_factor) else profit_factor,
        'ratio_avg_win_loss': NAN if (ratio_avg_win_loss != ratio_avg_win_loss) else ratio_avg_win_loss,
        'longs_count': NAN if (longs_count != longs_count) else longs_count,
        'longs_percentage': NAN if (longs_percentage != longs_percentage) else longs_percentage,
        'shorts_percentage': NAN if (shorts_percentage != shorts_percentage) else shorts_percentage,
        'shorts_count': NAN if (shorts_count != shorts_count) else shorts_count,
        'fee': NAN if (fee != fee) else fee,
        'net_profit': NAN if (net_profit != net_profit) else net_profit,
        'net_profit_percentage': NAN if (net_profit_percentage != net_profit_percentage) else net_profit_percentage,
        'average_win': NAN if (average_win != average_win) else average_win,
        'average_loss': NAN if (average_loss != average_loss) else average_loss,
        'expectancy': NAN if (expectancy != expectancy) else expectancy,
        'expectancy_percentage': NAN if (expectancy_percentage != expectancy_percentage) else expectancy_percentage,
        'expected_net_profit_every_100_trades': NAN if (
            expected_net_profit_every_100_trades != expected_net_profit_every_100_trades) else expected_net_profit_every_100_trades,
        'average_holding_period': NAN if (average_holding_period) != (average_holding_period) else (average_holding_period),
        'average_winning_holding_period': NAN if (average_winning_holding_period) != (average_winning_holding_period) else (average_winning_holding_period),
        'average_losing_holding_period': NAN if (average_losing_holding_period) != (average_losing_holding_period) else (average_losing_holding_period),
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'max_drawdown': NAN if (max_drawdown != max_drawdown) else max_drawdown,
        'annual_return': NAN if (annual_return != annual_return) else annual_return,
        'sharpe_ratio': NAN if (sharpe_ratio != sharpe_ratio) else sharpe_ratio,
        'calmar_ratio': NAN if (calmar_ratio != calmar_ratio) else calmar_ratio,
        'sortino_ratio': NAN if (sortino_ratio != sortino_ratio) else sortino_ratio,
        'omega_ratio': NAN if (omega_ratio != omega_ratio) else omega_ratio,
        'serenity_index': NAN if (serenity_index != serenity_index) else serenity_index,
        'smart_sharpe': NAN if (smart_sharpe != smart_sharpe) else smart_sharpe,
        'smart_sortino': NAN if (smart_sortino != smart_sortino) else smart_sortino,
        'total_open_trades': total_open_trades,
        'open_pl': open_pl,
        'winning_streak': winning_streak,
        'losing_streak': losing_streak,
        'largest_losing_trade': largest_losing_trade,
        'largest_winning_trade': largest_winning_trade,
        'current_streak': current_streak[-1],
        'kelly_criterion': kelly_criterion,
    }

def hyperparameters(routes_arr: list) -> list:
    if routes_arr[0].strategy.hp is None:
        return []
    hp = []
    # only for the first route
    for key in routes_arr[0].strategy.hp:
        hp.append([
            key, routes_arr[0].strategy.hp[key]
        ])
    return hp