# -*- coding: utf-8 -*-
from jesse.services.file import store_logs
import jesse.helpers as jh
from jesse.modes import backtest_mode
from jesse.config import config
from jesse.services import charts
from jesse.services import report
from jesse.routes import router
from datetime import datetime, timedelta
from jesse.store import store
import jesse.services.metrics as stats
from jesse.enums import trade_types

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

import pandas as pd
import numpy as np
import json 

from jesse.utils import numpy_candles_to_dataframe
from typing import Union
from jesse.services import report 

import codecs

from bottle import template
import os
import re

# from jesse.strategies import Strategy



from enum import IntEnum
class CD(IntEnum):
    date = 0
    open = 1
    close = 2
    high = 3
    low = 4
    volume = 5
    
def extract_function_names(file_path):
    pattern = r"ta\.(\w+)"  # Regular expression pattern to find 'ta.' followed by function names
    function_names = []

    with open(file_path, 'r') as file:
        file_contents = file.read()
        matches = re.findall(pattern, file_contents)  # Find all matches in the file
        function_names.extend(matches)
    function_names = [s.upper() for s in function_names]
    return function_names
    
def rename_duplicates(input_list):
    counts = {}
    new_list = []

    for item in input_list:
        if item in counts:
            counts[item] += 1
            new_item = f"{item} {counts[item]}"
        else:
            counts[item] = 1
            new_item = f"{item} 1" if input_list.count(item) > 1 else item
        
        new_list.append(new_item)

    return new_list
    
        
def pvsra(candles: np.ndarray, sequential: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    df = numpy_candles_to_dataframe(candles)
    average_volume = df["volume"].rolling(10).mean()
    climax = df["volume"] * (df["high"] - df["low"])
    highest_climax_10 = climax.rolling(window=10).max()

    df["averageVolume"] = average_volume
    df["climax"] = climax
    df["highestClimax10"] = highest_climax_10
    df['climaxVolume'] = ((df['volume'] >= 2 * average_volume) | (climax >= highest_climax_10)).astype(int)
    df['risingVolume'] = ((df['volume'] >= 1.5 * average_volume) & (df['climaxVolume'] != 1)).astype(int)
    df['isBull'] = (df["close"] > df["open"]).astype(int)
    df['risingBullVolume'] = ((df["risingVolume"] == 1) & (df["isBull"] == 1)).astype(int)
    df['risingBearVolume'] = ((df["risingVolume"] == 1) & (df["isBull"] == 0)).astype(int)
    df['climaxBullVolume'] = ((df["climaxVolume"] == 1) & (df["isBull"] == 1)).astype(int)
    df['climaxBearVolume'] = ((df["climaxVolume"] == 1) & (df["isBull"] == 0)).astype(int)

    return df if sequential else df.iloc[-1]


def generateReport(new_name: str = None, new_path: str = None, customData={}, chartConfig={}, indicators={}):

    chartConfig={'isPvsra':True}
    cdef np.ndarray candles 
    strategy = router.routes[0].strategy 
    
    metrics = report.portfolio_metrics()
    metrics = {k: str(v) for k, v in metrics.items()}
    metrics_json = json.dumps(metrics,indent=4)
    metrics_json = f'const metrics_data = {metrics_json}';
    
    file_name = jh.get_session_id() if not new_name else new_name
    studyname = backtest_mode._get_study_name()
    strategy_filepath = f'strategies/{studyname.split("-")[0]}/__init__.py'
    matching_indicator_names = list(extract_function_names(strategy_filepath))
    matching_list = []
    for match in matching_indicator_names:
        matching_list.append(ta_lib_functions[match])
    matching_list = rename_duplicates(matching_list)
    print(matching_list)
    if not config['env']['simulation']['preload_candles']:
        candles = store.candles.get_candles(router.routes[0].exchange, router.routes[0].symbol, router.routes[0].timeframe)
    else:
        candle_key = f'{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}'
        candles = store.candles.storage[candle_key].array[0:-1]

    indicator_key = f'{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}'
    slice_amount = strategy.slice_amount[indicator_key]
    candles = candles[slice_amount-1:]
    color_list = [
    "rgba(255, 99, 132, 0.7)",
    "rgba(54, 162, 235, 0.7)",
    "rgba(255, 206, 86, 0.7)",
    "rgba(75, 192, 192, 0.7)",
    "rgba(153, 102, 255, 0.7)",
    "rgba(255, 159, 64, 0.7)",
    "rgba(199, 199, 199, 0.7)",
    "rgba(163, 73, 164, 0.7)",
    "rgba(60, 179, 113, 0.7)",
    "rgba(255, 127, 80, 0.7)",
    "rgba(100, 149, 237, 0.7)",
    "rgba(218, 112, 214, 0.7)",
    "rgba(255, 228, 181, 0.7)",
    "rgba(32, 178, 170, 0.7)",
    "rgba(255, 69, 0, 0.7)",
    "rgba(173, 216, 230, 0.7)",
    "rgba(144, 238, 144, 0.7)",
    "rgba(255, 215, 0, 0.7)",
    "rgba(106, 90, 205, 0.7)",
    "rgba(240, 128, 128, 0.7)"
]
    # Custom Data
    first_candle_close = candles[0][2]
    for index, indicator in enumerate(indicators):
        unpacked = indicator[indicator_key]  # indicator data
        indicator_name = matching_list[index]
        isVisible = 'true' if abs(unpacked[0] - first_candle_close) <= first_candle_close * 0.5 else 'false'
        ind_dict = {"data": unpacked, "options": {"color": color_list[index]},"visible":isVisible}
        customData[indicator_name] = ind_dict

        
    tpl = r"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{title}}</title>
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
    />
    <link href="https://unpkg.com/tippy.js@6/dist/tippy.css" rel="stylesheet" />
    <link
      rel="stylesheet"
      href="https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css"
    />
    <style>
      #navi-button {
        width: 27px;
        height: 27px;
        position: absolute;
        display: none;
        padding: 7px;
        box-sizing: border-box;
        font-size: 10px;
        border-radius: 50%;
        text-align: center;
        z-index: 1000;
        color: #b2b5be;
        background: rgba(250, 250, 250, 0.95);
        box-shadow: 0 2px 5px 0 rgba(117, 134, 150, 0.45);
      }

      .utility-box {
        display: none;
        position: absolute;
        top: 40px; /* Adjusted this line. 40px accounts for the height of the button plus a margin */
        right: 9.8%; /* Adjusted this line to align with toggleUtilityBox */
        background-color: transparent;
        z-index: 1000; /* Make sure it's above other elements */
      }
      .utility-box button {
        display: block;
        margin: 5px 0;
      }

      #toggleUtilityBox {
        position: absolute;
        top: 10px;
        right: 9.8%;
        z-index: 3;
        background-color: rgba(
          100,
          100,
          100,
          0.7
        ); /* initial background color */
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s; /* smooth transitions */
      }

      #toggleUtilityBox:hover {
        background-color: rgba(
          80,
          80,
          80,
          0.8
        ); /* slightly darker background on hover */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3); /* subtle shadow */
        transform: scale(1.1); /* grow the icon a bit */
      }

      #toggleUtilityBox i {
        color: white; /* initial color of the cogwheel icon */
        font-size: 20px;
        transition: color 0.3s; /* smooth color transition */
      }

      #toggleUtilityBox:hover i {
        color: rgba(255, 255, 255, 0.9); /* slightly brighter icon on hover */
      }

      #toggleNightMode {
        background-color: rgba(100, 100, 100, 0.7);
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
        margin-top: 13px; /* to give some space between this button and the utility button */
      }

      #toggleNightMode:hover {
        background-color: rgba(80, 80, 80, 0.8);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
      }

      #toggleNightMode i {
        color: white;
        font-size: 20px;
        transition: color 0.3s;
      }

      #toggleNightMode:hover i {
        color: rgba(255, 255, 255, 0.9);
      }
      #resetScales {
        background-color: rgba(100, 100, 100, 0.7);
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
        margin-top: 6px; /* adjust as needed */
      }

      #resetScales:hover {
        background-color: rgba(80, 80, 80, 0.8);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
      }

      #resetScales i {
        color: white;
        font-size: 20px;
        transition: color 0.3s;
      }

      #resetScales:hover i {
        color: rgba(255, 255, 255, 0.9);
      }
      #resetButton {
        background-color: rgba(100, 100, 100, 0.7);
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
        margin-top: 6px; /* adjust as needed */
      }

      #resetButton:hover {
        background-color: rgba(80, 80, 80, 0.8);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
      }

      #resetButton i {
        color: white;
        font-size: 20px;
        transition: color 0.3s;
      }

      #resetButton:hover i {
        color: rgba(255, 255, 255, 0.9);
      }

      #toggleCandleColors {
        background-color: rgba(100, 100, 100, 0.7);
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
        margin-top: 6px; /* adjust as needed */
      }

      #toggleCandleColors:hover {
        background-color: rgba(80, 80, 80, 0.8);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
      }

      #toggleCandleColors i {
        color: white;
        font-size: 20px;
        transition: color 0.3s;
      }

      #toggleCandleColors:hover i {
        color: rgba(255, 255, 255, 0.9);
      }

      #metricsButton {
        background-color: rgba(100, 100, 100, 0.7);
        padding: 8px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
        margin-top: 6px; /* adjust as needed */
      }

      #metricsButton:hover {
        background-color: rgba(80, 80, 80, 0.8);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        transform: scale(1.1);
      }

      #metricsButton i {
        color: white;
        font-size: 20px;
        transition: color 0.3s;
      }

      #metricsButton {
        color: rgba(255, 255, 255, 0.9);
      }

      #metricsOverlay {
        position: absolute;
        top: 100px;
        left: 100px;
        width: 400px;
        background-color: white;
        border: 1px solid black;
        z-index: 1000;
        padding: 10px;
      }

      #metricsHeader {
        cursor: move;
        background-color: #f1f1f1;
        padding: 10px;
        border-bottom: 1px solid black;
      }

      #metricsContent {
        padding: 10px;
      }

      #closeMetricsButton {
        position: absolute; /* Position the button absolutely within its parent */
        top: 20px; /* Distance from the top of the container */
        right: 20px; /* Distance from the right of the container */
        cursor: pointer; /* Change cursor to pointer to indicate it's clickable */
        font-size: 15px; /* Adjust the size of the icon */
      }
      .metrics-table td:first-child {
         padding-right: 20px; /* Adjust padding as needed */
      }
    </style>
  </head>
  <body id="main-body" style="background-color: white">
    <button
      id="toggleUtilityBox"
      style="position: absolute; top: 10px; right: 10%; z-index: 3"
    >
      <i class="fas fa-cog"></i>
    </button>
    <div class="utility-box" id="utilityBox">
      <button id="toggleNightMode">
        <i class="fas fa-moon"></i>
      </button>
      <div id="resetScales">
        <svg width="20px" height="20px" viewBox="0 0 512 512" fill="#ffffff">
          <g id="SVGRepo_bgCarrier" stroke-width="0" />

          <g
            id="SVGRepo_tracerCarrier"
            stroke-linecap="round"
            stroke-linejoin="round"
          />

          <g id="SVGRepo_iconCarrier">
            <g
              id="Page-1"
              stroke="none"
              stroke-width="1"
              fill="none"
              fill-rule="evenodd"
            >
              <g
                id="Combined-Shape"
                fill="#ffffff"
                transform="translate(27.581722, 33.830111)"
              >
                <path
                  d="M100.418278,7.10542736e-15 L200.836556,100.418278 L170.666667,130.588167 L121.751,81.684 L121.751278,371.503223 L420.418278,371.503223 L420.418278,392.836556 L121.751278,392.836223 L121.751611,435.503223 L79.0849447,435.503223 L79.084278,392.836223 L36.418278,392.836556 L36.418278,371.503223 L79.084278,371.503223 L79.084,81.685 L30.1698893,130.588167 L7.10542736e-15,100.418278 L100.418278,7.10542736e-15 Z M364.598766,30.1698893 L364.599441,53.0058528 C375.002669,56.6829614 384.517019,62.2417388 392.733552,69.2732459 L412.530364,57.8432107 L442.159994,109.163235 L422.376075,120.586547 C423.349349,125.855024 423.858025,131.286386 423.858025,136.836556 C423.858025,142.386833 423.34933,147.818298 422.376019,153.086871 L442.159994,164.509877 L412.530364,215.829901 L392.733552,204.399866 C384.517019,211.431373 375.002669,216.990151 364.599441,220.667259 L364.598766,243.503223 L305.339506,243.503223 L305.338831,220.667259 C294.935988,216.990287 285.421958,211.431785 277.205634,204.400648 L257.407908,215.829901 L227.778278,164.509877 L247.562253,153.086871 C246.588942,147.818298 246.080247,142.386833 246.080247,136.836556 C246.080247,131.286035 246.588987,125.854336 247.562381,120.585546 L227.778278,109.163235 L257.407908,57.8432107 L277.20464,69.2733144 C285.421191,62.2417749 294.935569,56.6829733 305.338831,53.0058528 L305.339506,30.1698893 L364.598766,30.1698893 Z M334.969136,101.281 C315.332345,101.281 299.41358,117.199765 299.41358,136.836556 C299.41358,156.473347 315.332345,172.392112 334.969136,172.392112 C354.605927,172.392112 370.524691,156.473347 370.524691,136.836556 C370.524691,117.199765 354.605927,101.281 334.969136,101.281 Z"
                ></path>
              </g>
            </g>
          </g>
        </svg>
      </div>
      <button id="resetButton" title="Reset and Refresh">
        <i class="fas fa-redo"></i>
      </button>
      <button id="toggleCandleColors">
        <i class="fas fa-chart-line"></i>
      </button>
      <button id="metricsButton">
        <i class="fas fa-tachometer-alt"></i>
      </button>
    </div>
    <div id="metricsOverlay" style="display: none">
      <div id="metricsHeader" style="display: flex; justify-content: center; align-items: center;">
        <span style="flex-grow: 1; text-align: center; font-size: larger;"><strong>Performance Metrics</strong></span>
        <button id="closeMetricsButton" style="margin-left: auto;">
          <i class="fas fa-times"></i>
          <!-- Font Awesome close icon -->
        </button>
      </div>
      <div id="metricsContent"></div>
    </div>
    <div id="tvchart">

    </div>

  </body>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jscolor/2.5.1/jscolor.js"></script>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@popperjs/core/dist/umd/popper.min.js"></script>
  <script src="https://unpkg.com/tippy.js@6"></script>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.21/lodash.min.js"></script>

  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<script>
{{!candleData}}
{{!metrics}}
{{!orderData}}

    const processData = (data, mapper) => data.split('\n').map(mapper);
    let candleColorsEnabled = false; // By default, candlestick colors are not applied

    const candleMapper = (row) => {
        const [time, open, high, low, close, volume, color, borderColor, wickColor] = row.split(',');
        const res = {
            time: time * 1,
            open: open * 1,
            high: high * 1,
            low: low * 1,
            close: close * 1
        };
        if (candleColorsEnabled && color.trim() && borderColor.trim() && wickColor.trim()) {
            res.color = color;
            res.borderColor = borderColor;
            res.wickColor = wickColor;
        }
        return res;
    };

    const volumeMapper = (row) => {
        const [time, , , , , volume] = row.split(',');
        return {
            time: time * 1,
            value: volume * 1
        };
    };

    const orderMapper = (row) => {
        const [time, mode, side, type, qty, price, pnl_order] = row.split(',');
        return {
            time: time * 1,
            position: side === 'sell' ? 'aboveBar' : 'belowBar',
            color: side === 'sell' ? 'rgba(251, 192, 45, 1)' : '#2196F3',
            shape: side === 'sell' ? 'arrowDown' : 'arrowUp',
            text: `${type} @ ${price} : ${qty}${mode} $${Math.round(pnl_order * 100) / 100}`
        };
    };

    const pnlMapper = (row) => {
        const [time, , , , , , , pnl_accumulated] = row.split(',');
        return {
            time: time * 1,
            value: pnl_accumulated * 1
        };
    };

    // Data processing with error handling
    const getCandleData = async () => {
        try {
            return processData(candleData, candleMapper);
        } catch (error) {
            console.error("Error processing candle data:", error);
            throw new Error("Failed to process candle data.");
        }
    };

    const getVolumeData = async () => {
        try {
            return processData(candleData, volumeMapper);
        } catch (error) {
            console.error("Error processing volume data:", error);
            throw new Error("Failed to process volume data.");
        }
    };

    const getOrderData = async () => {
        try {
            return processData(orderData, orderMapper);
        } catch (error) {
            console.error("Error processing order data:", error);
            throw new Error("Failed to process order data.");
        }
    };

    const getPnlData = async () => {
        try {
            return processData(orderData, pnlMapper);
        } catch (error) {
            console.error("Error processing PnL data:", error);
            throw new Error("Failed to process PnL data.");
        }
    };

    const getCustomData = async (offset) => {
        try {
            const data = candleData.split('\n').map((row) => {
                const arr = row.split(',');
                return {
                    time: arr[0] * 1,
                    value: arr[offset + 9] * 1
                };
            });
            return data;
        } catch (error) {
            console.error("Error processing custom data:", error);
            throw new Error("Failed to process custom data.");
        }
    };

        var chartWidth = window.innerWidth-20;
        var chartHeight = window.innerHeight-20;
        const displayChart = async () => {
          const chartProperties = {
            width: chartWidth,
            height: chartHeight,
            handleScale: {
                mouseWheel: true,
                pinch: true,
            },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
            },
            layout: {
              background: {color: '#FFFFFF'},
              textColor: '#000000',
            },
            grid: {
              vertLines: {
                color: 'transparent',
              },
              horzLines: {
                color: 'transparent',
              },
            }, rightPriceScale:
              {		visible: true,
                  borderColor: 'rgba(197, 203, 206, 1)'	},
              leftPriceScale:
                { visible: true,
                  borderColor: 'rgba(197, 203, 206, 1)'	},
            timeScale: {
              timeVisible: true,
              secondsVisible: true,
            },
            crosshair: {
                // Change mode from default 'magnet' to 'normal'.
                // Allows the crosshair to move freely without snapping to datapoints
                mode: LightweightCharts.CrosshairMode.Normal,

                // Vertical crosshair line (showing Date in Label)
                vertLine: {
                    width: 6,
                    color: '#C3BCDB44',
                    style: LightweightCharts.LineStyle.Solid,
                    labelBackgroundColor: '#9B7DFF',
                },

                // Horizontal crosshair line (showing Price in Label)
                horzLine: {
                    color: '#9B7DFF',
                    labelBackgroundColor: '#9B7DFF',
                },
            },
          };


          const domElement = document.getElementById('tvchart');
          const chart = LightweightCharts.createChart(domElement, chartProperties);
          const candleseries = chart.addCandlestickSeries();
          const klinedata = await getCandleData();
          candleseries.setData(klinedata);
          const odata = await getOrderData();
          candleseries.setMarkers(odata);

          const volumeSeries = chart.addHistogramSeries({
                color: 'rgba(4, 111, 232, 0.25)',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: 'volume',
                scaleMargins: {
                    top: 0.7,
                    bottom: 0,
                },
            });
          const vdata = await getVolumeData();
          volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.7, // highest point of the series will be 70% away from the top
                bottom: 0,
            },
        });
          volumeSeries.setData(vdata);



          const pnl = chart.addLineSeries({color: 'rgba(93, 147, 187, 0.8)', lineWidth: 1, priceScaleId: 'left',})
          const pnl_data = await getPnlData()
          pnl.setData(pnl_data)

          //chart.timeScale().fitContent();

            //chart.timeScale().scrollToPosition(-20, false);

          // Create a button with common attributes
          function createNavigationButton(className, innerSVG, leftOffset, clickCallback) {
              const button = document.createElement('div');
              button.className = className;
              button.id = "navi-button"
              button.style.color = '#4c525e';
              button.innerHTML = innerSVG;

              button.addEventListener('click', clickCallback);

              button.addEventListener('mouseover', function() {
                  button.style.background = 'rgba(250, 250, 250, 1)';
                  button.style.color = '#000';
              });

              button.addEventListener('mouseout', function() {
                  button.style.background = 'rgba(250, 250, 250, 0.6)';
                  button.style.color = '#4c525e';
              });

              document.body.appendChild(button);
              return button;
          }

          const timeScale = chart.timeScale();

          // Button to go to the real time (end of the chart)
          const endButtonSVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M6.5 1.5l5 5.5-5 5.5M3 4l2.5 3L3 10"></path></svg>';
          const endButton = createNavigationButton('go-to-realtime-button', endButtonSVG, 60, () => {
              timeScale.scrollToRealTime();
          });

          // Button to go to the beginning of the chart
          const startButtonSVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M7.5 12.5l-5 -5.5 5 -5.5M11 10l-2.5 -3L11 4"></path></svg>';
          const startButton = createNavigationButton('go-to-beginning-button', startButtonSVG, 825, () => {
              timeScale.scrollToPosition(-(klinedata.length), true);
          });
            // Update the button positioning logic
            function updateButtonPositions() {
                chartWidth = window.innerWidth - 20;
                chartHeight = window.innerHeight - 20;

                startButton.style.left = '80px'; // Fixed position from left
                startButton.style.top = (chartHeight - 27 - 30) + 'px';

                endButton.style.right = '95px'; // Fixed position from right
                endButton.style.left = 'auto'; // Reset the left position in case it's set
                endButton.style.top = (chartHeight - 27 - 30) + 'px';
            }
            // Attach the resize event listener
            window.addEventListener('resize', updateButtonPositions);

            // Call the updateButtonPositions function to set initial positions
            updateButtonPositions();



          timeScale.subscribeVisibleTimeRangeChange(function() {
              const buttonVisible = timeScale.scrollPosition() < 0;
              endButton.style.display = buttonVisible ? 'block' : 'none';
              startButton.style.display = 'block';
          });

            // other functions:
          const dayProperties = {
          layout: {
            background: {color: '#FFFFFF'}, // White background for day mode
            textColor: '#000000',
          }};

          const nightProperties = {
          layout: {
            background: {color: '#000000'}, // Black background for night mode
            textColor: '#FFFFFF',
          }};


          // Tool Tip
          let symbolName = document.title.split('-')
          symbolName = symbolName[0] + ' ' + symbolName[1] + ' ' + symbolName[2]
          + '-' + symbolName[3] + ' ' + symbolName[4];
          const container = document.getElementById('tvchart');

          const legend = document.createElement('div');
          legend.style = `position: absolute; left: 8%; top: 0.5%; z-index: 6; font-size: 14px; font-family: sans-serif; line-height: 18px; font-weight: 300;`;
          legend.style.color = 'black';
          container.appendChild(legend);

          const getLastBar = series => {
            return series.dataByIndex(-1);
          };
        
       let seriesObjects = {}
        {{!customCharts}}


          const getMostRecentPnlBeforeTime = (pnl_data, timestamp) => {
          for (let i = pnl_data.length - 1; i >= 0; i--) {
              if (pnl_data[i].time <= timestamp) {
                  return pnl_data[i];
              }
          }
          return null;  // Return null if no such element is found
          };

          const symbol = document.title.split('-')[2];
          const setTooltipHtml = (name, date, open,close,high,low,volume,color,pnl,pnl_color,pnlicon,lineSeriesValues) => {
            const volumeVisible = volumeSeries.options().visible;
            const volumeVisible_icon = volumeVisible ? 'fas fa-eye' : 'fas fa-eye-slash';
            legend.innerHTML =
              `
                <div style="font-size: 19px; margin: 7px 0px 6px 0px;">
                ${name}</div>
                <div style="font-size: 16px; margin: 8px 0px;">
                  O <span style="color: ${color};">${open}</span>
                  H <span style="color: ${color};">${high}</span>
                  L <span style="color: ${color};">${low}</span>
                  C <span style="color: ${color};">${close}</span>
                </div>
                <div style="margin:8px 0px;">${date}</div>
                <div style="margin:8px 0px;">
                  <strong>Volume:</strong> ${volume} ${symbol}
                  <i class="fas fa-palette" id="volumeIcon" style="margin-left: 3px; cursor: pointer;"></i>
                  <input id="volumeColorPicker" class="jscolor" data-jscolor="{}" value="FF0000" style="display:none;">
                  <button id="toggleVolumeVisibility"><i id="clickable" class="${volumeVisible_icon}" style="cursor: pointer;"></i></button>
                </div>
                <div><strong>PnL:</strong>
                  <span style="color: ${pnl_color};">${pnl} </span> <i class="fas fa-palette" id="pnlIcon" style="margin-left: 3px; cursor: pointer;"></i>
                  <input id="pnlColorPicker" class="jscolor" data-jscolor="{}" value="FF0000" style="display:none;">
                  <button id="togglePnlVisibility"><i id="clickable" class="${pnlicon}" style="cursor: pointer;"></i></button>
                </div>

              `;
let lineSeriesHtml = '';
for (let data of lineSeriesValues) {
    const seriesName = data.name;
    const isVisible = seriesObjects[seriesName].options().visible;
    const visibilityIcon = isVisible ? 'fas fa-eye' : 'fas fa-eye-slash';

    lineSeriesHtml += `
    <div style="margin:8px 0px;">
        <strong>${seriesName}:</strong> ${data.value}
        <i class="fas fa-palette lineSeriesPaletteIcon" data-series-name="${seriesName}" style="margin-left: 3px; cursor: pointer;"></i>
        <input id="${seriesName}ColorPicker" class="lineSeriesColorPicker jscolor" data-jscolor="{}" value="FF0000" style="display:none;">
        <button class="toggleLineSeriesVisibility" data-series-name="${seriesName}"><i class="${visibilityIcon}" style="cursor: pointer;"></i></button>
    </div>`;
}

            legend.innerHTML += lineSeriesHtml;
          };

          const updateLegend = param => {
    let validCrosshairPoint = !(
        param === undefined || param.time === undefined || param.point.x < 0 || param.point.y < 0
    );
    let lineSeriesValues = [];
    let bar, time, volume, volume_value, color, mostRecentPnl, pnl_value, pnl_color, pnl_icon;

    bar = param.seriesData.get(candleseries);
    if (validCrosshairPoint && bar.time !== undefined) {
        let new_array = [...param.seriesData.entries()].slice(2);

        time = new Date(bar.time * 1000).toLocaleString('en', { timeZoneName: 'short' })
        volume = param.seriesData.get(volumeSeries);
        volume_value = volume.value.toFixed(1);
        color = bar.close > bar.open ? 'green' : 'red';
        mostRecentPnl = getMostRecentPnlBeforeTime(pnl_data, bar.time);
        pnl_value = mostRecentPnl ? mostRecentPnl.value.toFixed(1) : "N/A";
        pnl_color = pnl_value > 0 ? 'green' : 'red';
        pnl_icon = pnlVisible ? 'fas fa-eye' : 'fas fa-eye-slash';
        let lineSeriesCount = 0;
        const pnlValueFloat = parseFloat(pnl_value); // Convert pnl_value to a float for comparison
        let seriesKeys = Object.keys(seriesObjects);
        lineSeriesValues = new_array.filter(entry => {
            const valueFloat = parseFloat(entry[1].value.toFixed(2)); // Convert to a float for comparison
            const percentDifference = Math.abs((valueFloat - pnlValueFloat) / ((valueFloat + pnlValueFloat) / 2)) * 100;

            // Check if the value is not equal to pnl_value and has at least a 1 percent difference
            return valueFloat !== pnlValueFloat && percentDifference >= 1;
        }).map(entry => {
            let value = entry[1].value.toFixed(2);
            let seriesName = seriesKeys[lineSeriesCount++];
            return {name: seriesName, value: value};
        });

        // Save the last valid result to localStorage
        localStorage.setItem('lastValidLegendData', JSON.stringify({
            symbolName, volume_value,time,bar, color, pnl_value, pnl_color, pnl_icon, lineSeriesValues
        }));
    } else {
        // Retrieve the last valid result from localStorage
        const lastValidData = JSON.parse(localStorage.getItem('lastValidLegendData'));
        if (lastValidData) {
            ({ symbolName, volume_value,time,bar, color, pnl_value, pnl_color, pnl_icon, lineSeriesValues } = lastValidData);
        } else {
            // If no last valid data is found, return early
            return;
        }
    }

    setTooltipHtml(symbolName, time, bar.open.toFixed(1),
    bar.close.toFixed(1), bar.high.toFixed(1),
    bar.low.toFixed(1), volume_value, color, pnl_value, pnl_color, pnl_icon, lineSeriesValues);
};

          chart.subscribeCrosshairMove(updateLegend);

          let pnlVisible = true;

          document.body.addEventListener('click', function(event) {
          if (event.target.matches('#pnlIcon')) {
              jscolor.install();
              const pnlColorPicker = document.getElementById('pnlColorPicker');
              pnlColorPicker.jscolor.show();
              pnlColorPicker.onchange = function() {
                  const newColor = this.jscolor.toRGBString();
                  pnl.applyOptions({ color: newColor }); // Change the color of the PnL plot
                  localStorage.setItem('pnlColor', newColor); // Save the color in local storage
              };
             }
          else if (event.target.matches('#togglePnlVisibility') || event.target.closest('#togglePnlVisibility')) {
        pnlVisible = !pnlVisible;  // Toggle the visibility state
        pnl.applyOptions({ visible: pnlVisible });  // Set the visibility of the pnl series
              }
        // Handling the color picker display for line series
        else if (event.target.matches('.lineSeriesPaletteIcon')) {
            const seriesName = event.target.getAttribute('data-series-name');
            const colorPicker = document.getElementById(`${seriesName}ColorPicker`);
            jscolor.install();
            colorPicker.jscolor.show();
            colorPicker.onchange = function() {
                const newColor = this.jscolor.toRGBString();
                seriesObjects[seriesName].applyOptions({ color: newColor });
                localStorage.setItem(`${seriesName}Color`, newColor);
            };
        }

          // Handling the visibility toggle for line series
          else if (event.target.matches('.toggleLineSeriesVisibility') || event.target.closest('.toggleLineSeriesVisibility')) {
            // Your existing logic for toggling visibility
            const seriesName = event.target.closest('.toggleLineSeriesVisibility').getAttribute('data-series-name');
            const isVisible = seriesObjects[seriesName].options().visible;
            seriesObjects[seriesName].applyOptions({ visible: !isVisible });
          } else     if (event.target.matches('#volumeIcon')) {
            jscolor.install();
            const volumeColorPicker = document.getElementById('volumeColorPicker');
            volumeColorPicker.jscolor.show();
            volumeColorPicker.onchange = function() {
                const newColor = this.jscolor.toRGBString();
                volumeSeries.applyOptions({ color: newColor }); // Change the color of the volume series
                localStorage.setItem('volumeColor', newColor); // Save the color in local storage
            };
        } else if (event.target.matches('#toggleVolumeVisibility') || event.target.closest('#toggleVolumeVisibility')){
          const isVolumeVisible = volumeSeries.options().visible;
          volumeSeries.applyOptions({ visible: !isVolumeVisible });
          // Update the icon based on visibility
          const icon = document.querySelector('#toggleVolumeVisibility i');
          if (icon) {
              icon.className = isVolumeVisible ? 'fas fa-eye-slash' : 'fas fa-eye';
          }}



        });

          document.getElementById('toggleCandleColors').addEventListener('click', async function() {
              candleColorsEnabled = !candleColorsEnabled; // Toggle the state
              const klinedata = await getCandleData(); // Fetch the updated candle data
              candleseries.setData(klinedata); // Update the chart with the new data
          });


          let isNightMode = false; // Default to day mode
          document.getElementById('toggleNightMode').addEventListener('click', function() {
            const dn_iconElement = this.querySelector('i');
            if (isNightMode) {
              chart.applyOptions(dayProperties); // Apply day mode
              // Optionally, change other UI elements for day mode
              legend.style.color = 'black'
              document.getElementById('main-body').style.backgroundColor = '#FFFFFF'
              //volumeSeries.applyOptions({color: 'rgba(4, 111, 232, 0.2)'})
              dn_iconElement.className = 'fas fa-moon';
            } else {
              chart.applyOptions(nightProperties); // Apply night mode
              // Optionally, change other UI elements for night mode
              legend.style.color = 'white';
              document.getElementById('main-body').style.backgroundColor = '#000000'
              //volumeSeries.applyOptions({color: 'rgba(122, 168, 228, 0.55)' });
              dn_iconElement.className = 'fas fa-sun';
            }
            isNightMode = !isNightMode; // Toggle the mode
          });

            document.getElementById('resetScales').addEventListener('click', function() {
            const rightPriceScale = chart.priceScale('right');
            const leftPriceScale = chart.priceScale('left');
            chart.applyOptions({
              rightPriceScale: {autoScale: true},
              leftPriceScale: {autoScale: true}
            });

        });

          document.getElementById('toggleUtilityBox').addEventListener('click', function() {
            const utilityBox = document.getElementById('utilityBox');
            if (utilityBox.style.display === 'none' || utilityBox.style.display === '') {
                utilityBox.style.display = 'block';
            } else {
                utilityBox.style.display = 'none';
            }
        });


        document.addEventListener('keydown', function(event) {
        const timeScale = chart.timeScale();
        const currentPosition = timeScale.scrollPosition();
        const currentRange = timeScale.getVisibleLogicalRange();
        const rangeLength = currentRange.to - currentRange.from;
        const zoomFactor = 0.2;
        let newRangeLength;

        switch(event.code) {
            // Left arrow key
            case 'ArrowLeft':
                timeScale.scrollToPosition(currentPosition - 1000, false);
                return;  // Exit the function early after handling the arrow key

            // Right arrow key
            case 'ArrowRight':
                timeScale.scrollToPosition(currentPosition + 1000, false);
                return;  // Exit the function early after handling the arrow key

            case 'Equal':  // This corresponds to the "+" key
                newRangeLength = rangeLength * (1 - zoomFactor);
                break;

            case 'Minus':  // This corresponds to the "-" key
                newRangeLength = rangeLength * (1 + zoomFactor);
                break;

            default:
                return;  // If any other key is pressed, exit the function early
        }

          const newFrom = currentRange.from + (rangeLength - newRangeLength) / 2;
          const newTo = newFrom + newRangeLength;

          timeScale.setVisibleLogicalRange({
              from: newFrom,
              to: newTo
          });
      });

        // Event Listener for Reset Button
        document.getElementById('resetButton').addEventListener('click', function() {
            localStorage.clear();  // Clear local storage
            location.reload();    // Refresh the page
        });

    // Step 1: Refactor the Tippy initialization into a separate function
    function initializeTippy() {
        tippy('#utilityBox button i, #utilityBox div svg', {
            content: function(reference) {
                let parent = reference.parentElement;
                switch(parent.id || parent.parentElement.id) {
                    case 'toggleNightMode':
                      return isNightMode ? 'Day Mode' : 'Night Mode';
                    case 'resetScales':
                        return 'Reset Y-Axis Scales';
                    case 'resetButton':
                        return 'Reset Settings';
                    case 'toggleCandleColors':
                        return candleColorsEnabled ? 'Default Candle Colors' : 'PVRSA Candle Colors';
                    case 'metricsButton':
                        return 'Strategy Metrics';
                    default:
                        return ''; // default tooltip content
                }
            },
            interactive: false,
            placement: 'left',
            arrow: true,
            delay: [50, 25],
            animation: 'scale'
        });
    }

    // Step 2: Use the function in the DOMContentLoaded event
    document.addEventListener('DOMContentLoaded', function() {
        initializeTippy();

        // Step 3: Attach click event listeners to the utility buttons
        let utilityButtons = document.querySelectorAll('#utilityBox button, #utilityBox div');
        utilityButtons.forEach(button => {
            button.addEventListener('click', initializeTippy);
        });
    });

        // Helper function to round values
        const roundValue = (value, precision = 2) => {
            return _.round(value, precision);
        };

        // Helper function to convert seconds to a human-readable format
        const secondsToHumanReadable = (seconds) => {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds - (hours * 3600)) / 60);
            const secondsLeft = roundValue(seconds - (hours * 3600) - (minutes * 60));
            return `${hours}h ${minutes}m ${secondsLeft}s`;
        };

        const populateMetricsOverlay = () => {
          let contentHtml = '<table class="metrics-table">'; // Add a class for easier targeting

          let startingTimestamp = (klinedata[0].time)*1000
          let endingTimestamp = (klinedata[klinedata.length - 1].time)*1000
          // Convert timestamps to Date objects
          let startingDate = new Date(startingTimestamp);
          let endingDate = new Date(endingTimestamp);
          // Format dates to "yyyy-mm-dd"
          let formatDate = (date) => date.toISOString().split('T')[0];

          let formattedStartingDate = formatDate(startingDate);
          let formattedEndingDate = formatDate(endingDate);
          // Calculate the difference in days
          const oneDay = 24 * 60 * 60 * 1000; // milliseconds in a day
          const diffDays = Math.round(Math.abs((endingDate - startingDate) / oneDay));

          // Calculate the difference in years
          const diffYears = diffDays / 365.25; // considering leap years

          // Format the trading period
          const tradingPeriod = `${diffDays} days (${diffYears.toFixed(2)} years)`;

          // Construct the table rows based on the metrics data
          const metrics = [
              ['Trading Period', tradingPeriod],
              ['Starting Date', formattedStartingDate],
              ['Ending Date', formattedEndingDate],
              ['Total Closed Trades', metrics_data.total],
              ['Total Net Profit', `${roundValue(metrics_data.net_profit)} (${roundValue(metrics_data.net_profit_percentage)}%)`],
              ['Starting => Finishing Balance', `${roundValue(metrics_data.starting_balance)} => ${roundValue(metrics_data.finishing_balance)}`],
              ['Open Trades', metrics_data.total_open_trades],
              ['Total Paid Fees', roundValue(metrics_data.fee)],
              ['Max Drawdown', `${roundValue(metrics_data.max_drawdown)}%`],
              ['Annual Return', `${roundValue(metrics_data.annual_return)}%`],
              ['Expectancy', `${roundValue(metrics_data.expectancy)} (${roundValue(metrics_data.expectancy_percentage)}%)`],
              ['Avg Win | Avg Loss', `${roundValue(metrics_data.average_win)} | ${roundValue(metrics_data.average_loss)}`],
              ['Ratio Avg Win / Avg Loss', roundValue(metrics_data.ratio_avg_win_loss)],
              ['Win-rate', `${roundValue(metrics_data.win_rate * 100)}%`],
              ['Profit Factor', roundValue(metrics_data.profit_factor)],
              ['Longs | Shorts', `${roundValue(metrics_data.longs_percentage)}% | ${roundValue(metrics_data.shorts_percentage)}%`],
              ['Avg Holding Time', secondsToHumanReadable(metrics_data.average_holding_period)],
              ['Winning Trades Avg Holding Time', secondsToHumanReadable(metrics_data.average_winning_holding_period)],
              ['Losing Trades Avg Holding Time', secondsToHumanReadable(metrics_data.average_losing_holding_period)],
              ['Sharpe Ratio', roundValue(metrics_data.sharpe_ratio)],
              ['Calmar Ratio', roundValue(metrics_data.calmar_ratio)],
              ['Sortino Ratio', roundValue(metrics_data.sortino_ratio)],
              ['Omega Ratio', roundValue(metrics_data.omega_ratio)],
              ['Winning Streak', metrics_data.winning_streak],
              ['Losing Streak', metrics_data.losing_streak],
              ['Largest Winning Trade', roundValue(metrics_data.largest_winning_trade)],
              ['Largest Losing Trade', roundValue(metrics_data.largest_losing_trade)],
              ['Total Winning Trades', metrics_data.total_winning_trades],
              ['Total Losing Trades', metrics_data.total_losing_trades],
              ['Kelly Criterion', roundValue(metrics_data.kelly_criterion)],
          ];

            metrics.forEach(([label, value]) => {
                contentHtml += `<tr><td>${label}</td><td>${value}</td></tr>`;
            });

            contentHtml += '</table>';

            document.getElementById('metricsContent').innerHTML = contentHtml;
              $('#metricsOverlay').draggable({ handle: "#metricsHeader" }).resizable({
                  resize: function() {
                      adjustFontSize();
                  }
              }).show();
              adjustFontSize(); // Adjust font size on initial display
          };

          document.getElementById('metricsButton').addEventListener('click', populateMetricsOverlay);

          document.getElementById('closeMetricsButton').addEventListener('click', function() {
              document.getElementById('metricsOverlay').style.display = 'none';
          });

          function adjustFontSize() {
              const metricsContent = document.getElementById('metricsContent');
              const metricsTable = metricsContent.querySelector('.metrics-table');
              const containerWidth = metricsContent.offsetWidth;

              // Calculate the font size based on the container width
              const fontSize = Math.max(containerWidth / 25, 10); // Example calculation, adjust as needed

              metricsTable.style.fontSize = fontSize + 'px';
          }



        function updateWindowSize() {
            chartWidth = window.innerWidth-20;
            chartHeight = window.innerHeight-20;
          chart.applyOptions({     width: chartWidth,    height: chartHeight, });

        }

          window.onresize = updateWindowSize;

            // populate data from localstorage
            const savedPnlColor = localStorage.getItem('pnlColor');
          if (savedPnlColor) {
            // If there's a saved color in local storage, apply it
            pnl.applyOptions({ color: savedPnlColor });
           }
          // Iterate through the seriesObjects to apply saved colors
          for (let seriesName in seriesObjects) {
              const savedColor = localStorage.getItem(`${seriesName}Color`);
              if (savedColor) { // If a color was saved in local storage for this series
                  seriesObjects[seriesName].applyOptions({ color: savedColor });
              }
          }


        };


        displayChart();
  </script>
</html>
      
        
      
        """

        # if('mainChartLines' in customData and len(customData['mainChartLines'])>0):
        #   for key, value in customData['mainChartLines'].items():
        #   #   candles += [value]
        #     for idx, item in enumerate(candles):
        #       item = np.append(item, value[idx])
        
        
        

    pvsraData = pvsra(candles, True)

    candleDataList = []

    for idx, candle in enumerate(candles):
        # Extract commonly used data
        current_date = str(candle[CD.date] / 1000)
        current_open = str(candle[CD.open])
        current_high = str(candle[CD.high])
        current_low = str(candle[CD.low])
        current_close = str(candle[CD.close])
        current_volume = str(candle[CD.volume])


        color_data = [',gray,gray,gray,']  # default
        if pvsraData['climaxBullVolume'].iloc[idx] == 1:
            color_data = [',lime,lime,white,']
        elif pvsraData['climaxBearVolume'].iloc[idx] == 1:
            color_data = [',red,red,gray,']
        elif pvsraData['risingBullVolume'].iloc[idx] == 1:
            color_data = [',blue,blue,white,']
        elif pvsraData['risingBearVolume'].iloc[idx] == 1:
            color_data = [',fuchsia,fuchsia,gray,']
        elif pvsraData['isBull'].iloc[idx] == 1:
            color_data = [',silver,silver,gray,']


        custom_data_list = []
        if customData:
            for key, value in customData.items():
                if len(value['data']) > idx:
                    custom_data_list.append(f"{(value['data'][idx])}")
                else:
                    custom_data_list.append("")

        # Build the complete candle data string
        candleDataList.append(
            f"{current_date},{current_open},{current_high},{current_low},{current_close},{current_volume}"
            f"{''.join(color_data)}"
            f"{','.join(custom_data_list)}"
        )

    # Join all the candle data strings
    candleData = '\n'.join(candleDataList)
    candleData = f'const candleData = `{candleData}`'

    if candleData[-1] == '\n':
        candleData = candleData[:-1]  # remove last new line
    candleData += ';'

    pnl_accumulated = 0
    orderDataList = ['const orderData = `']

    # Extract common configuration once
    config_prefix = 'env.exchanges.'
    trades = store.completed_trades.trades

    for trade in trades:
        trading_fee = jh.get_config(f'{config_prefix}{trade.exchange}.fee')
        average_entry_price = 0
        average_entry_size = 0
        side_factor = 1 if trade.type != trade_types.SHORT else -1

        for order in trade.orders:
            if not order.is_executed:
                continue

            fee = abs(order.qty) * order.price * trading_fee
            is_buy_for_long = trade.type == trade_types.LONG and order.side == 'buy'
            is_sell_for_short = trade.type == trade_types.SHORT and order.side == 'sell'

            if is_buy_for_long or is_sell_for_short:
                pnl_order = -fee
                total_price = average_entry_price * average_entry_size + order.price * abs(order.qty)
                total_qty = average_entry_size + abs(order.qty)
                average_entry_price = total_price / total_qty
                average_entry_size = total_qty
            else:
                pnl_order = (order.price - average_entry_price) * abs(order.qty) * side_factor - fee
                average_entry_size -= abs(order.qty)

            pnl_accumulated += pnl_order

            mode = ''
            if order.is_stop_loss:
                mode = ' (SL)'
            elif order.is_take_profit:
                mode = ' (TP)'

            orderDataList.append(f"{order.executed_at/1000},{mode},{order.side},{order.type},{order.qty},{order.price},{pnl_order},{pnl_accumulated},{trade.id}\n")

    # Join list to form the final string
    orderData = ''.join(orderDataList).rstrip('\n') + '`;'
            
    customCharts = ''

    if len(customData) > 0:
        idx = 0
        for key, value in customData.items():
            if 'options' not in value:
                value['options'] = {}
            value['options']['title'] = key
            if 'type' not in value:
                value['type'] = 'LineSeries'
            
            isVisible = value['visible']

            cstLineTpl = f"""
                const series{idx+1} = chart.add{value['type']}({str(value['options'])});
                series{idx+1}.setData(await getCustomData({idx}));
                seriesObjects['{key}'] = series{idx+1};
                seriesObjects['{key}'].applyOptions({{ visible: {isVisible} }});
            """
            
            customCharts += cstLineTpl
            idx += 1


        

    # pnlCharts = ''
    # priceScale = ''
    # pnlCharts = 'chart.addLineSeries({color: \'rgba(4, 111, 232, 1)\', lineWidth: 1, priceScaleId: \'left\',}).setData(await getPnlData())'
    # priceScale = ' rightPriceScale: {		visible: true, borderColor: \'rgba(197, 203, 206, 1)\'	}, leftPriceScale: { visible: true, borderColor: \'rgba(197, 203, 206, 1)\'	},'

    info = {'title': studyname,
        'candleData': candleData,
        'orderData': orderData,
        'customCharts':customCharts,
        # 'pnlCharts':pnlCharts,
        # 'priceScale': priceScale,
        'metrics': metrics_json
        }
        
    result = template(tpl, info)

    filename = "storage/JesseTradingViewLightReport/" + file_name + '.html'
    if new_path:
        filename= new_path + file_name + ' Interactive Chart .html'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with codecs.open(filename, "w", "utf-8") as f:
        f.write(result)
        
    if not new_path:
        import webbrowser
        url = f'C:/Python39/Algotrading/{filename}'
        webbrowser.open(url, new=2)  # open in new tab
    else:
        return filename


ta_lib_functions = {
    "AD": "Chaikin A/D Line",
    "ADOSC": "Chaikin A/D Oscillator",
    "ADX": "Average Directional Movement Index",
    "ADXR": "Average Directional Movement Index Rating",
    "APO": "Absolute Price Oscillator",
    "AROON": "Aroon",
    "AROONOSC": "Aroon Oscillator",
    "ATR": "Average True Range",
    "AVGPRICE": "Average Price",
    "BBANDS": "Bollinger Bands",
    "BETA": "Beta",
    "BOP": "Balance Of Power",
    "CCI": "Commodity Channel Index",
    "CDL2CROWS": "Two Crows",
    "CDL3BLACKCROWS": "Three Black Crows",
    "CDL3INSIDE": "Three Inside Up/Down",
    "CDL3LINESTRIKE": "Three Outside Up/Down",
    "CDL3STARSINSOUTH": "Three Stars In The South",
    "CDL3WHITESOLDIERS": "Three Advancing White Soldiers",
    "CDLABANDONEDBABY": "Abandoned Baby",
    "CDLADVANCEBLOCK": "Advance Block",
    "CDLBELTHOLD": "Belt-hold",
    "CDLBREAKAWAY": "Breakaway",
    "CDLCLOSINGMARUBOZU": "Closing Marubozu",
    "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
    "CDLCOUNTERATTACK": "Counterattack",
    "CDLDARKCLOUDCOVER": "Dark Cloud Cover",
    "CDLDOJI": "Doji",
    "CDLDOJISTAR": "Doji Star",
    "CDLDRAGONFLYDOJI": "Dragonfly Doji",
    "CDLENGULFING": "Engulfing Pattern",
    "CDLEVENINGDOJISTAR": "Evening Doji Star",
    "CDLEVENINGSTAR": "Evening Star",
    "CDLGAPSIDESIDEWHITE": "Up/Down-gap side-by-side white lines",
    "CDLGRAVESTONEDOJI": "Gravestone Doji",
    "CDLHAMMER": "Hammer",
    "CDLHANGINGMAN": "Hanging Man",
    "CDLHARAMI": "Harami Pattern",
    "CDLHARAMICROSS": "Harami Cross Pattern",
    "CDLHIGHWAVE": "High-Wave Candle",
    "CDLHIKKAKE": "Hikkake Pattern",
    "CDLHIKKAKEMOD": "Modified Hikkake Pattern",
    "CDLHOMINGPIGEON": "Homing Pigeon",
    "CDLIDENTICAL3CROWS": "Identical Three Crows",
    "CDLINNECK": "In-Neck Pattern",
    "CDLINVERTEDHAMMER": "Inverted Hammer",
    "CDLKICKING": "Kicking",
    "CDLKICKINGBYLENGTH": "Kicking - bull/bear determined by the longer marubozu",
    "CDLLADDERBOTTOM": "Ladder Bottom",
    "CDLLONGLEGGEDDOJI": "Long Legged Doji",
    "CDLLONGLINE": "Long Line Candle",
    "CDLMARUBOZU": "Marubozu",
    "CDLMATCHINGLOW": "Matching Low",
    "CDLMATHOLD": "Mat Hold",
    "CDLMORNINGDOJISTAR": "Morning Doji Star",
    "CDLMORNINGSTAR": "Morning Star",
    "CDLONNECK": "On-Neck Pattern",
    "CDLPIERCING": "Piercing Pattern",
    "CDLRICKSHAWMAN": "Rickshaw Man",
    "CDLRISEFALL3METHODS": "Rising/Falling Three Methods",
    "CDLSEPARATINGLINES": "Separating Lines",
    "CDLSHOOTINGSTAR": "Shooting Star",
    "CDLSHORTLINE": "Short Line Candle",
    "CDLSPINNINGTOP": "Spinning Top",
    "CDLSTALLEDPATTERN": "Stalled Pattern",
    "CDLSTICKSANDWICH": "Stick Sandwich",
    "CDLTAKURI": "Takuri (Dragonfly Doji with very long lower shadow)",
    "CDLTASUKIGAP": "Tasuki Gap",
    "CDLTHRUSTING": "Thrusting Pattern",
    "CDLTRISTAR": "Tristar Pattern",
    "CDLUNIQUE3RIVER": "Unique 3 River",
    "CDLUPSIDEGAP2CROWS": "Upside Gap Two Crows",
    "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods",
    "CMO": "Chande Momentum Oscillator",
    "CORREL": "Pearson's Correlation Coefficient ®",
    "DEMA": "Double Exponential Moving Average",
    "DX": "Directional Movement Index",
    "EMA": "Exponential Moving Average",
    "HT_DCPERIOD": "Hilbert Transform - Dominant Cycle Period",
    "HT_DCPHASE": "Hilbert Transform - Dominant Cycle Phase",
    "HT_PHASOR": "Hilbert Transform - Phasor Components",
    "HT_SINE": "Hilbert Transform - SineWave",
    "HT_TRENDLINE": "Hilbert Transform - Instantaneous Trendline",
    "HT_TRENDMODE": "Hilbert Transform - Trend vs Cycle Mode",
    "KAMA": "Kaufman Adaptive Moving Average",
    "LINEARREG": "Linear Regression",
    "LINEARREG_ANGLE": "Linear Regression Angle",
    "LINEARREG_INTERCEPT": "Linear Regression Intercept",
    "LINEARREG_SLOPE": "Linear Regression Slope",
    "MA": "All Moving Average",
    "MACD": "Moving Average Convergence/Divergence",
    "MACDEXT": "MACD with controllable MA type",
    "MACDFIX": "Moving Average Convergence/Divergence Fix 12/26",
    "MAMA": "MESA Adaptive Moving Average",
    "MAX": "Highest value over a specified period",
    "MAXINDEX": "Index of highest value over a specified period",
    "MEDPRICE": "Median Price",
    "MFI": "Money Flow Index",
    "MIDPOINT": "MidPoint over period",
    "MIDPRICE": "Midpoint Price over period",
    "MIN": "Lowest value over a specified period",
    "MININDEX": "Index of lowest value over a specified period",
    "MINMAX": "Lowest and highest values over a specified period",
    "MINMAXINDEX": "Indexes of lowest and highest values over a specified period",
    "MINUS_DI": "Minus Directional Indicator",
    "MINUS_DM": "Minus Directional Movement",
    "MOM": "Momentum",
    "NATR": "Normalized Average True Range",
    "OBV": "On Balance Volume",
    "PLUS_DI": "Plus Directional Indicator",
    "PLUS_DM": "Plus Directional Movement",
    "PPO": "Percentage Price Oscillator",
    "ROC": "Rate of change : ((price/prevPrice)-1)*100",
    "ROCP": "Rate of change Percentage: (price-prevPrice)/prevPrice",
    "ROCR": "Rate of change ratio: (price/prevPrice)",
    "ROCR100": "Rate of change ratio 100 scale: (price/prevPrice)*100",
    "RSI": "Relative Strength Index",
    "SAR": "Parabolic SAR",
    "SAREXT": "Parabolic SAR - Extended",
    "SMA": "Simple Moving Average",
    "STDDEV": "Standard Deviation",
    "STOCH": "Stochastic",
    "STOCHF": "Stochastic Fast",
    "STOCHRSI": "Stochastic Relative Strength Index",
    "SUM": "Summation",
    "T3": "Triple Exponential Moving Average (T3)",
    "TEMA": "Triple Exponential Moving Average",
    "TRANGE": "True Range",
    "TRIMA": "Triangular Moving Average",
    "TRIX": "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
    "TSF": "Time Series Forecast",
    "TYPPRICE": "Typical Price",
    "ULTOSC": "Ultimate Oscillator",
    "VAR": "Variance",
    "WCLPRICE": "Weighted Close Price",
    "WILLR": "Williams' %R",
    "WMA": "Weighted Moving Average"
}