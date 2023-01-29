This is a maintained modified version of the Jesse backtesting framework. Created with the intention of maximizing backtesting speed and introducing support for stock and forex backtesting/optimization. Currently, it is still **under development**. 

## Planned New Features

* Forex backtesting/optimization 

## New Features

* Stock backtesting/optimization 
* Optional indicator precalculation w/o candle preloading
* Monte Carlo Simulation
* Significantly improved backtest simulation speed


## Removed Features

* live trading

## Installation 

First, install the necessary python packages listed in the "requirements.txt" via the pip package manager. Then execute the cythonize.py script in the root of where you downloaded this repository. Copy the "jesse" folder and place it where you downloaded the origional Jesse repository. 


## Benchmark

Iteration times for multiple timeframes were recorded for a backtest using the example [SMACrossover](https://github.com/jesse-ai/example-strategies/blob/master/SMACrossover/__init__.py) strategy on the Bitfinex exchange with the BTC-USD pair from 2018-01-01 - 2022-11-01
##### Original 

```bash
5m : 71.56 seconds
15m : 57.25 seconds
30m : 52.78 seconds
45m : 50.77 seconds
1h : 51.09 seconds
2h : 49.87 seconds
3h : 49.15 seconds
4h : 49.81 seconds
```

##### Optimized (Without Indicator Precalculation)

```bash
5m : 7.92 seconds
15m : 3.11 seconds
30m : 1.74 seconds
45m : 1.29 seconds
1h : 1.13 seconds 
2h : 0.71 seconds
3h : 0.60 seconds
4h : 0.57 seconds
```

##### Optimized (With Indicator Precalculation and Preloaded Candles)

```bash
5m : 2.56 seconds
15m : 0.95 seconds 
30m : 0.51 seconds
45m : 0.39 seconds
1h : 0.31 seconds
2h : 0.21 seconds
3h : 0.17 seconds
4h : 0.16 seconds
```

## Acknowledgements

 - [Jesse](https://github.com/jesse-ai/jesse)