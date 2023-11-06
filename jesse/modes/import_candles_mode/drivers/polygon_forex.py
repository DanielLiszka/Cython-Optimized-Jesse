from polygon.rest.client import RESTClient
import jesse.helpers as jh
from jesse.models import Candle
from pandas import * 
from datetime import date, datetime, timedelta
from typing import Any, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter 
from urllib3.util.retry import Retry
import time
import os 

class Polygon_Forex(RESTClient):
    def __init__(self, auth_key: str=['api_key'], timeout:int=5):
        super().__init__(auth_key)
        retry_strategy = Retry(total=15,
                               backoff_factor=10,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount('https://', adapter)

    def get_tickers(self, market:str=None) -> list:

        resp = self.reference_tickers_v3(market=market)
        if hasattr(resp, 'results'):
            df = pd.DataFrame(resp.results)

            while hasattr(resp, 'next_url'):
                resp = self.reference_tickers_v3(next_url=resp.next_url)
                df = df.append(pd.DataFrame(resp.results))

            # if market == 'crypto':
                # Only use USD pairings.
            df = df[df['currency_symbol'] == 'USD']
            df['name'] = df['base_currency_name']
            df = df[['ticker', 'name', 'market', 'active']]

            df = df.drop_duplicates(subset='ticker')
            df = df['ticker'].values.tolist()
            df1 = [x for x in df if str(x) != 'nan']
            return df1
        return None  

    def get_bars(self, market:str=None, ticker:str=None, multiplier:int=1,
                 timespan:str='minute', from_:date=None, to:date=None) -> None:

        payload = {
            'unadjusted': 'false',
            'sort': 'asc',
            'limit': 50000,
        }
        old_ticker = ticker
        ticker = f'C:{ticker}USD'
        if ticker is None:
            raise Exception('Ticker must not be None.')
        if ticker not in (self.get_tickers('fx')):
            raise Exception(f'{ticker} is unavailable')    
        from_ = from_ if from_ else date(2000,1,1)
        to = date.today()

        resp = self.forex_currencies_aggregates(ticker, multiplier, timespan,
                                      from_.strftime('%Y-%m-%d'), to.strftime('%Y-%m-%d'),
                                      **payload)
        df = pd.DataFrame(resp.results)
        last_minute = 0
        while resp.results[-1]['t'] > last_minute:
            last_minute = resp.results[-1]['t']
            last_minute_date = datetime.fromtimestamp(last_minute/1000).strftime('%Y-%m-%d')
            resp = self.forex_currencies_aggregates(ticker, multiplier, timespan,
                                      last_minute_date, to.strftime('%Y-%m-%d'),
                                      **payload)
            new_bars = pd.DataFrame(resp.results)
            df = df.append(new_bars[new_bars['t'] > last_minute])
        if not os.path.exists('storage/temp/forex bars'):
            os.makedirs('storage/temp/forex bars')
        try:
            old_df = pd.read_csv(f'storage/temp/forex bars/{ticker}.csv')
            old_df_lasttime = old_df['t'].iloc[-1]
            old_df_firsttime = old_df['t'].iloc[0]
            old_df = old_df.append(df[df['t'] > old_df_lasttime])
            prepend_data = df[df['t'] < old_df_firsttime]
            old_df = pd.concat([prepend_data, old_df]).sort_values(by='t').reset_index(drop=True)
            df = old_df
            ticker = old_ticker
            df.to_csv(f'storage/temp/forex bars/{ticker}.csv', sep=',', index=False) 
        except FileNotFoundError:
            try: 
                df.to_csv(f'storage/temp/forex bars/{ticker}.csv', sep=',', index=False) 
            except:
                raise Exception('Error occured while saving stock data')  
        
        



