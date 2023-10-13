from numba import njit 
import numpy as np
from scipy.stats import norm 
from jesse.utils import numpy_candles_to_dataframe
import pandas as pd 
import random
#for stock handling 
# @njit
# def generate_candle_from_one_minutes(timeframe: str,
                                     # candles: np.ndarray ,
                                     # accept_forming_candles:bool = False) :
    # return np.array([
        # candles[0][0],
        # candles[0][1] if not np.isnan(candles[:, 1]).any() else np.nan,
        # candles[-1][2] if not np.isnan(candles[:, 2]).any() else np.nan,
        # candles[:, 3].max() if not np.isnan(candles[:, 3]).any() else np.nan,
        # candles[:, 4].min() if not np.isnan(candles[:, 4]).any() else np.nan,
        # candles[:, 5].sum() if not np.isnan(candles[:, 5]).any() else np.nan,
    # ],dtype= np.float64)
    
@njit
def fast_convert(nump: np.ndarray ):
    open = nump[:,2]
    close = nump[:,3]
    # high = nump{:,4]
    # low = nump[:,5]
    # time = nump[:,8]
    # volume = nump[:,0]
    for i in range(nump.shape[0]):
        if not np.isnan(close[i]) and np.isnan(open[i]):
            nump[i][2] = close[i-1]
            nump[i][4] = close[i]
            nump[i][5] = close[i]
            nump[i][0] = 0         
    return np.vstack((
        nump[:,8],
        nump[:,2],
        nump[:,3], 
        nump[:,4],
        nump[:,5], 
        nump[:,0])).T
    
def stock_candles_func(ticker, start_date, finish_date,exchange): 	
    if exchange == 'Polygon_Stocks':
        #Example: AAPL-USD
        ticker = ticker.split('-')[0]
        df = pd.read_csv(f'storage/temp/stock bars/{ticker}.csv')
    else:
        ticker = ticker.split('-')[0]
        #Example: AED-USD
        # ticker = f'C:{ticker}USD'
        df = pd.read_csv(f'storage/temp/forex bars/{ticker}.csv')
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df['date_index'] = df['date']
    df = df.set_index('date_index')
    df = df.asfreq(freq='1T')
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq = '1T')
    df = df.reindex(idx, method= 'pad')
    df = df.rename(columns={'o':'open',
        'c':'close',
        'h':'high',
        'l':'low',
        'v':'volume',
        'vw':'vwap',
        'n':'transactions'})
    df[['close','vwap']] = df[['close','vwap']].fillna(method='ffill')
    df[['volume']] = df[['volume']].fillna(value=0)
    df['date'] = df.index
    df['date'] = (df['date'].astype('int64')//1e9)*1000
    mask = (df['date'] >= start_date) & (df['date'] <= finish_date)
    df_slice = df.loc[mask]
    # print(f'DF Date: {df_slice}')
    
    nump = df_slice.to_numpy()
    nump = nump.astype(np.float64)
    # for i in range(9):
        # print(f'Column Number: {i} - {nump[:,i]}')
    # pd.DataFrame(nump).to_csv(f'in{random.random()}.csv')
    out = fast_convert(nump) 
    out = np.array(out,dtype= np.float64)
    # print(f'start_date: {start_date} - finish_date: {finish_date}')
    # pd.DataFrame(out).to_csv(f'out{random.random()}.csv')
    # pd.DataFrame(nump).to_csv('nump.csv')
    # print(df.head())
    return out

@njit
def generate_candle_from_one_minutes(timeframe: str,
                                     candles ,
                                     accept_forming_candles:bool = False) :
    return np.array([
        candles[0][0],
        candles[0][1],
        candles[-1][2], 
        candles[:, 3].max(),
        candles[:, 4].min(), 
        candles[:, 5].sum(), 
    ],dtype= np.float64)
    
   
def get_nan_indices(array):
    return np.where(np.isnan(array))[0]


def reinsert_nan(array, nan_indices):
    for i in range(nan_indices.shape[0]):
        array = np.concatenate((array[:nan_indices[i]], [np.nan], array[nan_indices[i]:]))
    return array
    
# def monte_carlo_candles(candles):
    # import pandas as pd 
    # import pandas_montecarlo 
    # import random 
    # randomnum = (random.randint(2,100))
    # df = numpy_candles_to_dataframe(candles)
    # close = df['close'].montecarlo(sims=100)
    # close = pd.DataFrame(close.data[[randomnum]])
    # open = df['open'].montecarlo(sims=100)
    # open = pd.DataFrame(open.data[[randomnum]])
    # low = df['low'].montecarlo(sims=100)
    # low = pd.DataFrame(low.data[[randomnum]])
    # high = df['high'].montecarlo(sims=100)
    # high = pd.DataFrame(high.data[[randomnum]])
    # volume = df['volume'].montecarlo(sims=100)
    # volume = pd.DataFrame(volume.data[[randomnum]])

    # close = close.to_numpy()
    # high = high.to_numpy()
    # low = low.to_numpy()
    # open = open.to_numpy()
    # volume = volume.to_numpy()
    # print(close[-10:])
    
    # open = np.ravel(open[:,0])
    # close = np.ravel(close[:,0])
    # high = np.ravel(high[:,0])
    # low = np.ravel(low[:,0])

    # return  open, close, high, low 
    
#works but not true montecarlo simulation 
def monte_carlo_candles(candles: np.array) -> np.array:
    log_returns = (np.log(1 + (np.diff(candles) / candles[:-1] * 100)))
    u = (np.nanmean(log_returns))
    var = (np.nanvar(log_returns))
    drift = u - (0.5 * var)
    stdev = np.nanstd(log_returns)
    rando = np.random.rand(len(candles))
    daily_returns = np.exp(drift + stdev + rando) #1+((np.exp((drift) + (stdev) * norm.ppf(rando)*1.5))*.0001-.0001)  #
    price_list = np.zeros_like(daily_returns)
    price_list[0] = candles[0]
    for d in range(1, len(candles)):
        price_list[d] = price_list[d - 1] * daily_returns[d]    
    print(price_list)
    return price_list


 