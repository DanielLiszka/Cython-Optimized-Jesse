# from polygon.rest.client import RESTClient
from requests import HTTPError
# from .stock_data_importing import MyRESTClient
import jesse.helpers as jh
from .interface import CandleExchange
import datetime
import pandas as pd 
import numpy as np 

class Polygon_Stocks(CandleExchange):

    def __init__(self):
        super().__init__('Polygon', 5000, 0.01, stock_mode=True,backup_exchange_class=None)
        
    def init_backup_exchange(self):
        self.backup_exchange = None

    def get_starting_time(self, symbol):
        return None

