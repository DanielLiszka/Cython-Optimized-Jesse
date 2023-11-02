This is a maintained modified version of the Jesse backtesting framework. Created with the intention of maximizing backtesting speed and introducing support for stock and forex backtesting/optimization.

 Built-in methods were added for pregenerating candles and precalculating indicators to improve speed wihtout affecting accuracy. If the indicators are precalculated then the last candle close is used for higher timeframe indicator calculations rather than the current partially formed candle. It's possible to backtest stocks, forex, and cryptocurrency concurrently if the indicators are precalculated and the candles are not pregenerated. 

## New Features

* Stock backtesting/optimization 
* Forex backtesting/optimization
* Optional indicator precalculation w/o candle preloading
* Monte Carlo Simulation
* Significantly improved backtest simulation speed
* Polygon.io stock and forex candle importing driver

## Removed Features

* live trading

## Installation 

First, install the necessary python packages listed in the "requirements.txt" via the pip package manager. Then execute the cythonize.py script in the root of where you downloaded this repository. Copy the "jesse" folder and place it where you downloaded the origional Jesse repository. 


## Benchmark

Iteration times for multiple timeframes were recorded for a backtest using the example [SMACrossover](https://github.com/jesse-ai/example-strategies/blob/master/SMACrossover/__init__.py) strategy on the Bitfinex exchange with the BTC-USD pair from 2018-01-01 - 2023-09-01
##### Original 

```bash
3m : 96.39 seconds 
5m : 77.32 seconds
15m : 61.08 seconds
30m : 57.41 seconds
45m : 56.57 seconds
1h : 54.30 seconds
2h : 53.50 seconds
3h : 52.93 seconds
4h : 52.23 seconds
```

##### Optimized (Without Indicator Precalculation)

```bash
3m : 14.91 seconds 
5m : 8.54 seconds
15m : 3.03 seconds
30m : 1.59 seconds
45m : 1.09 seconds
1h : 0.86 seconds 
2h : 0.49 seconds
3h : 0.38 seconds
4h : 0.33 seconds
```

##### Optimized (With Indicator Precalculation and Preloaded Candles)

```bash
3m : 3.25 seconds
5m : 2.01 seconds
15m : 0.74 seconds 
30m : 0.43 seconds
45m : 0.32 seconds
1h : 0.26 seconds
2h : 0.17 seconds
3h : 0.14 seconds
4h : 0.12 seconds
```

## Acknowledgements

 - [Jesse](https://github.com/jesse-ai/jesse)