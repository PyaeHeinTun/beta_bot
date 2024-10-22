import asyncio
import time
import pandas as pd
from beta_bot.helper import base as helper
from beta_bot.controller import base as controller
from datetime import datetime, timedelta
from beta_bot.database import base as database
from beta_bot.temp_storage import temp_storage_data, TempStorage
import ccxt.pro as ccxt

async def watch_ohlcv_info(
    exchange: ccxt.binance, symbol, timeframe, limit
) -> pd.DataFrame:
    config = temp_storage_data[TempStorage.config]
    is_dry_run = config["dry_run"]
    if(is_dry_run):
        data = await exchange.watch_ohlcv(
            symbol,
            timeframe,
            limit=limit,
            params={
                "enableRateLimit": True,
                "price": "mark",
            },
        )
    else:
        data = await exchange.watch_ohlcv(
            symbol,
            timeframe,
            limit=limit,
            params={
                "enableRateLimit": True,
            },
        )
    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    return df


async def fetch_ohlcv_info(
    exchange: ccxt.binance, symbol, timeframe, limit
) -> pd.DataFrame:
    # Fetch 10,000 candles in chunks of 1500 candles
    num_candles_to_fetch = limit
    candles_per_request = 1000
    start_index = 0
    current_time = datetime.strptime(
        str(datetime.utcnow()).split(".")[0], "%Y-%m-%d %H:%M:%S"
    )

    dataframe = pd.DataFrame()

    while start_index < num_candles_to_fetch:
        # Determine the end index for this request
        end_index = min(start_index + candles_per_request, num_candles_to_fetch)

        minute_data = (num_candles_to_fetch / end_index) * candles_per_request
        previous_time_utc = str(current_time - timedelta(minutes=minute_data))
        formatted_datetime_str = previous_time_utc.split(".")[0]
        # 2024-04-15 19:52:00
        since = exchange.parse8601(formatted_datetime_str)
        # round(datetime.strptime(formatted_datetime_str, '%Y-%m-%d %H:%M:%S').timestamp())
        # if (end_index == (num_candles_to_fetch-candles_per_request)) else None
        # Fetch candles for this chunk

        config = temp_storage_data[TempStorage.config]
        is_dry_run = config["dry_run"]
        if (is_dry_run):
            candles = await exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=candles_per_request,
                since=since,
                params={"price": "mark"},
            )
        else:
            candles = await exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=candles_per_request,
                since=since,
            )

        # Convert candles to DataFrame
        candles_df = pd.DataFrame(
            candles, columns=["date", "open", "high", "low", "close", "volume"]
        )
        candles_df["date"] = pd.to_datetime(candles_df["date"], unit="ms")
        # Append the candles to the DataFrame
        dataframe = pd.concat([dataframe, candles_df], ignore_index=True)

        # # Update start index for the next request
        start_index += candles_per_request
    return dataframe


async def fetch_data(exchange: ccxt.binance, symbol, timeframe, limit):
    cursor, conn = database.connectDB()
    temp_storage_data[TempStorage.cursor] = cursor
    temp_storage_data[TempStorage.conn] = conn
    temp_storage_data[TempStorage.exchange] = exchange
    current_trade_list = []

    while True:
        _command_for_run = temp_storage_data[TempStorage.command_for_run]
        if _command_for_run == "start":
            speed_counter_start = datetime.now()
            current_time = datetime.utcnow()

            # Initial Fetch Data
            if ((current_time.minute in [0])) or (
                f"order_book{symbol}" not in temp_storage_data
            ):
                temp_storage_data[TempStorage.should_count_for_current][symbol] = False
                temp_storage_data[TempStorage.dataframe][symbol] = (
                    await fetch_ohlcv_info(exchange, symbol, timeframe, limit)
                )
            temp_storage_data[TempStorage.current_data][symbol] = (
                await watch_ohlcv_info(exchange, symbol, timeframe, limit)
            )
            temp_storage_data[f"order_book{symbol}"] = await exchange.watch_order_book(
                symbol
            )

            # When New Data is arrived add to the existing dataframe
            isDateChangeFromPrevious = (
                temp_storage_data[TempStorage.dataframe][symbol].iloc[-1]["date"]
                != temp_storage_data[TempStorage.current_data][symbol].iloc[-1]["date"]
            )

            if (
                isDateChangeFromPrevious
                and temp_storage_data[TempStorage.should_count_for_current][symbol]
            ):
                last_index = len(temp_storage_data[TempStorage.dataframe][symbol])
                temp_storage_data[TempStorage.dataframe][symbol].loc[last_index] = (
                    temp_storage_data[TempStorage.current_data][symbol].iloc[-1]
                )
                temp_storage_data[TempStorage.dataframe][symbol].drop(
                    temp_storage_data[TempStorage.dataframe][symbol].index[0],
                    inplace=True,
                )
                temp_storage_data[TempStorage.dataframe][symbol].reset_index(
                    drop=True, inplace=True
                )
                temp_storage_data[TempStorage.future_prediction][symbol] = (
                    controller.predict_future(
                        temp_storage_data[TempStorage.dataframe][symbol],
                        symbol,
                    )
                )
                temp_storage_data[TempStorage.should_count_for_current][symbol] = False

            else:
                temp_storage_data[TempStorage.dataframe][symbol].iloc[-1] = (
                    temp_storage_data[TempStorage.current_data][symbol].iloc[-1]
                )
                temp_storage_data[TempStorage.should_count_for_current][symbol] = True

            # If Not Predicted, Then Predict
            if not symbol in temp_storage_data[TempStorage.future_prediction]:
                temp_storage_data[TempStorage.future_prediction][symbol] = (
                    controller.predict_future(
                        temp_storage_data[TempStorage.dataframe][symbol],
                        symbol,
                    )
                )

            dataframe = temp_storage_data[TempStorage.dataframe][symbol]
            # Strategy Manager
            trade_datas = await controller.main_trade_func(dataframe, temp_storage_data[TempStorage.future_prediction], symbol)
            current_trade_list = trade_datas
            speed_counter_stop = datetime.now()
            helper.logger(current_trade_list, speed_counter_start, speed_counter_stop)
        else:
            speed_counter_start = datetime.now()
            if f"order_book{symbol}" in temp_storage_data:
                temp_storage_data.pop(f"order_book{symbol}")
            speed_counter_stop = datetime.now()
            helper.logger(current_trade_list, speed_counter_start, speed_counter_stop)
            await asyncio.sleep(0.5)


async def fetch_multiple_coins():
    config = temp_storage_data[TempStorage.config]
    is_dry_run = config["dry_run"]
    
    exchange = ccxt.binance()
    if (is_dry_run):
        exchange.set_sandbox_mode(True)
        exchange.api = config["exchange"]["test_key"]
        exchange.apiKey = config["exchange"]["test_key"]
        exchange.secret = config["exchange"]["test_secret"]
    else:
        exchange.api = config["exchange"]["real_key"]
        exchange.apiKey = config["exchange"]["real_key"]
        exchange.secret = config["exchange"]["real_secret"]

    leverage = config["trade_params"]["leverage"]
    exchange.options["defaultType"] = config["exchange"]["type"]
    timeframe = config["timeframe"]
    limit = config["exchange"]["ohlcv_candle_limit"]
    symbols = config["exchange"]["pair_whitelist"]

    tasks = []
    for symbol in symbols:
        await exchange.set_leverage(leverage=leverage, symbol=symbol)
        temp_storage_data[TempStorage.dataframe][symbol] = pd.DataFrame()
        task = asyncio.create_task(fetch_data(exchange, symbol, timeframe, limit))
        tasks.append(task)

    await asyncio.gather(*tasks)