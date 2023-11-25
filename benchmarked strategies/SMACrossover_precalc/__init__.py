"""
Simple Moving Average Crossover Strategy
Author: FengkieJ (fengkiejunis@gmail.com)
Simple moving average crossover strategy is the ''hello world'' of algorithmic trading.
This strategy uses two SMAs to determine '''Golden Cross''' to signal for long position, and '''Death Cross''' to signal for short position.
"""

from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils

class SMACrossover_precalc(Strategy):

    def hyperparameters(self):
        return [
        {'name': 'Fast SMA', 'type': int, 'min': 12, 'max': 260, 'default': 50, 'step':1}, 
        {'name': 'Slow SMA', 'type': int, 'min': 8, 'max': 240, 'default': 200, 'step':1}, 
        ]
        
    def _indicator1(self,precalc_candles = None,sequential=False):
        return ta.sma(precalc_candles, period=self.hp['Fast SMA'],sequential=sequential)
        
    @property
    def _indicator1_test(self):
        return self._indicator1(precalc_candles=self.candles)
        
    def _indicator2(self,precalc_candles = None,sequential=False):
        return ta.sma(precalc_candles, period=self.hp['Slow SMA'],sequential=sequential)
        
    @property
    def _indicator2_test(self):
        return self._indicator2(precalc_candles=self.candles)
        
    def _indicator3(self,precalc_candles=None,sequential=False):
        return None 
    def _indicator4(self,precalc_candles=None,sequential=False):
        return None
    def _indicator5(self,precalc_candles=None,sequential=False):
        return None
    def _indicator6(self,precalc_candles=None,sequential=False):
        return None
    def _indicator7(self,precalc_candles=None,sequential=False):
        return None
    def _indicator8(self,precalc_candles=None,sequential=False):
        return None
    def _indicator9(self,precalc_candles=None,sequential=False):
        return None
    def _indicator10(self,precalc_candles=None,sequential=False):
        return None

    def should_long(self) -> bool:
        # Golden Cross (reference: https://www.investopedia.com/terms/g/goldencross.asp)
        # Fast SMA above Slow SMA
        return self._indicator1_value[-1] > self._indicator2_value[-1]

    def should_short(self) -> bool:
        # Death Cross (reference: https://www.investopedia.com/terms/d/deathcross.asp)
        # Fast SMA below Slow SMA
        return self._indicator1_value[-1] < self._indicator2_value[-1]

    def should_cancel_entry(self) -> bool:
        return False

    def go_long(self):
        # Open long position and use entire balance to buy
        qty = utils.size_to_qty(self.balance*0.1, self.price, fee_rate=self.fee_rate)

        self.buy = qty, self.price

    def go_short(self):
        # Open short position and use entire balance to sell
        qty = utils.size_to_qty(self.balance*0.1, self.price, fee_rate=self.fee_rate)

        self.sell = qty, self.price

    def update_position(self):
        # If there exist long position, but the signal shows Death Cross, then close the position, and vice versa.
        if self._indicator1_value[-1] < self._indicator2_value[-1] and self.is_long:
            self.liquidate()
    
        if self._indicator1_value[-1] > self._indicator2_value[-1] and self.is_short:
            self.liquidate()
            