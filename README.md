This is a maintained modified version of the Jesse backtesting framework. Created with the intention of maximizing backtesting speed and introducing support for stock and forex backtesting/optimization. 


## New Features

* Stock backtesting/optimization 
* Forex backtesting/optimization
* Optional indicator precalculation w/o candle preloading
* Monte Carlo Simulation
* Significantly improved backtest simulation speed
* Polygon.io stock data candle driver

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
3m : 15.54 seconds 
5m : 9.37 seconds
15m : 3.28 seconds
30m : 1.68 seconds
45m : 1.18 seconds
1h : 0.93 seconds 
2h : 0.56 seconds
3h : 0.43 seconds
4h : 0.36 seconds
```

##### Optimized (With Indicator Precalculation and Preloaded Candles)

```bash
3m : 4.01 seconds 
5m : 2.48 seconds
15m : 0.87 seconds 
30m : 0.48 seconds
45m : 0.37 seconds
1h : 0.29 seconds
2h : 0.19 seconds
3h : 0.16 seconds
4h : 0.14 seconds
```

## Acknowledgements

 - [Jesse](https://github.com/jesse-ai/jesse)