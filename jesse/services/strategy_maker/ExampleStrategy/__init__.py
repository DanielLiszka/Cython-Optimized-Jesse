from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils


class ExampleStrategy(Strategy):

    def _indicator1(self,precalc_candles = None):
        return None
   
    @property
    def _indicator1_test(self):
        return None
        
    def _indicator2(self,precalc_candles = None):
        return None
   
    @property
    def _indicator2_test(self):
        return None
               
    def _indicator3(self,precalc_candles=None):
        return None
   
    @property
    def _indicator3_test(self):
        return None
      
    def _indicator4(self, precalc_candles=None):
        return None
            
    @property
    def _indicator4_test(self):
        return None

    def _indicator5(self, precalc_candles=None):
        return None
    
    @property
    def _indicator5_test(self):
        return None

    def _indicator6(self,precalc_candles=None):
        return None
        
    @property 
    def _indicator6_test(self):
        return None

    def _indicator7(self,precalc_candles=None):
        return None
        
    @property
    def _indicator7_test(self):
        return None
    
    def _indicator8(self, precalc_candles=None):
        return None
        
    @property
    def _indicator8_test(self):
        return None
        
    def _indicator9(self, precalc_candles=None):
        return None
        
    @property
    def _indicator9_test(self):
        return None
        
    def _indicator10(self, precalc_candles=None):
        return None
        
    @property
    def _indicator10_test(self):
        return None

    def should_long(self) -> bool:
        return False

    def should_short(self) -> bool:
        # For futures trading only
        return False

    def should_cancel_entry(self) -> bool:
        return True

    def go_long(self):
        pass

    def go_short(self):
        pass
