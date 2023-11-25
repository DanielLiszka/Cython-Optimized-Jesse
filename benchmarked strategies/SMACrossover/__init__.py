"""
Simple Moving Average Crossover Strategy
Author: FengkieJ (fengkiejunis@gmail.com)
Simple moving average crossover strategy is the ''hello world'' of algorithmic trading.
This strategy uses two SMAs to determine '''Golden Cross''' to signal for long position, and '''Death Cross''' to signal for short position.
"""

from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils

class SMACrossover(Strategy):

    def hyperparameters(self):
        return [
        {'name': 'sma_period1', 'type': int, 'min': 6, 'max': 240, 'default': 200, 'step':1}, 
        {'name': 'sma_period2', 'type': int, 'min': 6, 'max': 260, 'default': 50, 'step':1}, 
        ]
    @property
    def slow_sma(self):
        return ta.sma(self.candles, self.hp['sma_period1'])

    @property
    def fast_sma(self):
        return ta.sma(self.candles, self.hp['sma_period2'])

    def should_long(self) -> bool:
        # Golden Cross (reference: https://www.investopedia.com/terms/g/goldencross.asp)
        # Fast SMA above Slow SMA
        return self.fast_sma > self.slow_sma

    def should_short(self) -> bool:
        # Death Cross (reference: https://www.investopedia.com/terms/d/deathcross.asp)
        # Fast SMA below Slow SMA
        return self.fast_sma < self.slow_sma

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
        if self.is_long and self.fast_sma < self.slow_sma:
            self.liquidate()
    
        if self.is_short and self.fast_sma > self.slow_sma:
            self.liquidate()
            
