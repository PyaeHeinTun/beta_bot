from datetime import datetime
from prettytable import PrettyTable
import os


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal


def logger(trade_list: list, startTime: datetime, endTime: datetime):
    clear_terminal()  # Clear the terminal before printing the table
    table = PrettyTable()
    table.field_names = ["id", "pair", "mode", "entry_price",
                         "current_price", "pnl", "roi", "ms"]
    table.float_format = f'.{6}'

    if (len(trade_list) == 0):
        table.add_row(["0", "Searching", "N", "0.00", "0.00", "0.00",
                       "0.00", round((endTime-startTime).microseconds * 0.001, 2)])
    else:
        for trade in trade_list:
            profit = trade.calculate_profit_ratio()
            id = str(trade.id)
            pair = trade.pair
            mode = "S" if trade.is_short == 1 else "L"
            entry_price = trade.entry_price
            current_price = trade.exit_price
            pnl = profit['pnl']
            roi = profit['roi']
            ms = round((endTime-startTime).microseconds * 0.001, 2)
            table.add_row([id, pair, mode, entry_price,
                          current_price, pnl, roi, ms])
    table.align = "r"
    table_string = table.get_string()
    print(table_string)
    print(f"updated_at: {datetime.utcnow()}", end="",
          flush=True)


def findMaxNumberOfKeys(future_predictions: dict) -> str:
    max_keys = []
    max_length = 0

    # Iterate over each key-value pair in the dictionary
    for key, value in future_predictions.items():
        count = sum(1 for k in value.keys()
                    if 0 <= k <= 60)
        if count > max_length:
            max_keys = [key]
            max_length = count
        elif count == max_length:
            max_keys.append(key)
    return max_keys[0]


def logger_test(future_predictions: dict, startTime: datetime, endTime: datetime):
    clear_terminal()  # Clear the terminal before printing the table
    table = PrettyTable()

    symbol_list = list(future_predictions.keys())  # Symbol List
    minute_list = list(future_predictions[findMaxNumberOfKeys(
        future_predictions)].keys())

    table.field_names = ["symbol"] + minute_list + ["ms"]
    table.float_format = f'.{6}'

    for symbol in symbol_list:
        coin_data: dict = future_predictions[symbol]
        class_prediction = [coin_data[minute]['class'] if (
            coin_data.keys().__contains__(minute)) else 0 for minute in minute_list]
        table.add_row(
            [symbol] + class_prediction + [round((endTime-startTime).microseconds * 0.001, 2)])

    table.align = "r"
    table_string = table.get_string()
    print(table_string)
    print(f"updated_at: {datetime.utcnow()}", end="",
          flush=True)
