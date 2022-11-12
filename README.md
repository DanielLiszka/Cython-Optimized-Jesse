This is a maintained modified version of the Jesse backtesting framework. Created with the intention of maximizing backtesting speed and introducing support for stock and forex backtesting/optimization. Currently, it is still under development. 

## Planned New Features

* Stock backtesting/optimization 
* Forex backtesting/optimization 
* optional indicator precalculation (initially hardcoded)
* Monte Carlo Simulation
* Significantly improved backtest simulation speed

## Removed Features

* live trading


## Benchmark

Iteration times were recorded for a backtest using the example [SMACrossover](https://github.com/jesse-ai/example-strategies/blob/master/SMACrossover/__init__.py) strategy on the Bitfinex exchange with the BTC-USD pair from 2018-01-01 - 2022-11-01
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

##### Optimized (With Indicator Precalculation)

```bash
5m : 5.40 seconds
15m : 2.08 seconds 
30m : 1.22 seconds
45m : 0.97 seconds
1h : 0.85 seconds
2h : 0.63 seconds
3h : 0.56 seconds
4h : 0.53 seconds
```

## Acknowledgements

 - [Jesse](https://github.com/jesse-ai/jesse)