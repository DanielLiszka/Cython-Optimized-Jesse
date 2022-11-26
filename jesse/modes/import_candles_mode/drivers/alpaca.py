from alpaca_trade_api.rest import TimeFrame, URL, REST
from requests import HTTPError
import jesse.helpers as jh
from .interface import CandleExchange


class Alpaca(CandleExchange):

    def __init__(self):
        super().__init__('Alpaca', 1000, 3, stock_mode=True,backup_exchange_class=None)
        try:
            api_key = jh.get_config('env.exchanges.Alpaca.api_key')
        except:
            raise ValueError("Alpaca api_key missing in config.py")
        api_key_id = jh.get_config('env.exchanges.Alpaca.api_key')
        api_secret = jh.get_config('env.exchanges.Alpaca.secret_key')
        base_url = "https://paper-api.alpaca.markets"
        self.restclient = REST(key_id=api_key_id,secret_key=api_secret,base_url=URL(base_url))
        

    def init_backup_exchange(self):
        self.backup_exchange = None

    def get_starting_time(self, symbol):

        return None

    def fetch(self, symbol, start_timestamp):

        base = jh.base_asset(symbol)
        # Check if symbol exists. Raises HTTP 404 if it doesn't.
        # try:
            # details = self.restclient.reference_ticker_details(base)
        # except HTTPError:
            # raise ValueError("Symbol ({}) probably doesn't exist.".format(base))

        # payload = {
            # 'unadjusted': 'false',
            # 'sort': 'asc',
            # 'limit': self.count,
        # }

        # Alpaca takes string dates not timestamps
        start =  jh.timestamp_to_date(start_timestamp)
        end = jh.timestamp_to_date(start_timestamp + (self.count) * 60000)
        response = self.restclient.get_barset(symbols = base, timeframe = TimeFrame.Minute , limit = 1000,start = start, end = end)

        data = response
        
        candles = []

        for d in data:
            candles.append({
                'id': jh.generate_unique_id(),
                'symbol': symbol,
                'exchange': self.name,
                'timestamp': int(d['t']),
                'open': float(d['o']),
                'close': float(d['c']),
                'high': float(d['h']),
                'low': float(d['l']),
                'volume': int(d['v'])
            })

        return candles