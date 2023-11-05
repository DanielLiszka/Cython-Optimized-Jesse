import os
import time
import sys
from config import config
import numpy as np
import itertools
from jesse import research
import json
import datetime
from pathlib import Path
from jesse.helpers import date_to_timestamp

def generateCorrelationTable(exchange,start_date,finish_date,timeframe):
    start = date_to_timestamp(start_date)
    finish = date_to_timestamp(finish_date)
    if exchange in ['Polygon_Stocks','Polygon_Forex']:
        print('Stocks or Forex are not supported yet')
        return
        # try:
            # pairs = get_stocks_with_enough_candles(exchange,start,finish)
        # except Exception as e:
            # print(e)
    else:
        try: 
            pairs = get_crypto_with_enough_candles(exchange,start,finish)
        except Exception as e:
            print(e)
            
    count = len(pairs)
    grid = np.ones(shape=(count, count))

    for (i, first), (j, second) in itertools.combinations(enumerate(pairs), 2):
        try:
            first_candles = research.get_candles(exchange, first, timeframe, start_date, finish_date)
            second_candles = research.get_candles(exchange, second, timeframe, start_date, finish_date)
        except Exception as e:
            print("An exception occurred")
            print(e)
            break

        res = np.corrcoef(first_candles[:, -1], second_candles[:, -1])

        grid[i, j], grid[j, i] = res[0, 1], res[1, 0]

    metadata = {
        "pairs": list(map(lambda item: item.replace('-', '/'), pairs)),
        "exchange": exchange,
        "timeframe": timeframe,
        "start_date": start_date.replace('-', '.'),
        "finish_date": finish_date.replace('-', '.'),
    }

    #
    # Generate html output
    #
    
    result = {'grid': grid.tolist(), 'metadata': metadata}
    result = json.dumps(result)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(script_dir, 'sample.html')
    with open(html_file_path, 'r') as file:
        file_data = file.read()
    
    title = "Correlation Table ◾ {}".format(datetime.datetime.today().strftime("%d %b, %Y %H:%M"))

    file_data = file_data \
        .replace('#DATE', title) \
        .replace('#JSON', result)

    Path("./storage/correlations").mkdir(parents=True, mode=0o777, exist_ok=True)

    file_name = "storage/correlations/correlation__{}.html".format(datetime.datetime.today().strftime("%Y-%m-%d__%H-%M"))

    with open(file_name, 'w') as file:
        file.write(file_data)
        
    import webbrowser
    working_path = wsl_path_to_windows(os.getcwd())
    url = f'{working_path}/{file_name}'
    webbrowser.open(url, new=2) 

def get_crypto_with_enough_candles(exchange_name: str, start_timestamp: str, end_timestamp: str) -> list:
    
    from jesse.services.db import database
    # SQL query
    sql_query = f"""
    SELECT DISTINCT symbol
    FROM candle as main
    WHERE timestamp BETWEEN '{start_timestamp}' AND '{end_timestamp}'
    AND exchange = '{exchange_name}'
    AND EXISTS (
        SELECT 1 FROM candle as before
        WHERE before.symbol = main.symbol
        AND before.exchange = main.exchange
        AND before.timestamp < '{start_timestamp}'
    )
    AND EXISTS (
        SELECT 1 FROM candle as after
        WHERE after.symbol = main.symbol
        AND after.exchange = main.exchange
        AND after.timestamp > '{end_timestamp}'
    )
    ORDER BY symbol;
    """

    symbols_with_enough_candles = []

    # Open the database connection if it's not already open
    if database.is_closed():
        database.open_connection()

    # Execute the query
    with database.db.atomic() as transaction:
        try:
            cursor = database.db.execute_sql(sql_query)
            symbols_with_enough_candles = cursor.fetchall()
        except Exception as e:
            print(f"An error occurred: {e}")
            transaction.rollback()
        finally:
            cursor.close()

    # Close the database connection
    database.close_connection()

    symbols_with_enough_candles = [item[0] for item in symbols_with_enough_candles]
    return symbols_with_enough_candles

def get_stocks_with_enough_candles(exchange,start_date,finish_date):
    pass
    

def wsl_path_to_windows(wsl_path):
    if wsl_path.startswith("/mnt/"):
        path_without_mnt = wsl_path[5:]
        drive_letter = path_without_mnt[0]
        rest_of_path = path_without_mnt[2:]
        windows_path = f"{drive_letter.upper()}:/{rest_of_path}"
        # windows_path = windows_path.replace('/', '\\')
        return windows_path
    else:
        raise wsl_path