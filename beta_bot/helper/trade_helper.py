from datetime import datetime, timedelta
from decimal import Decimal


def is_quarter_hour(time: datetime):
    seconds = time.second
    return (seconds <= 30) & (seconds >= 10)


# def calculate_deviation(num1, num2):
#     diff = abs(Decimal(str(num1)) - Decimal(str(num2)))
#     diff_str = str(diff).lstrip('0').replace('.', '')
#     deviation_result = int(diff_str)
#     return deviation_result


def calculate_deviation(num1, num2, order_book):
    bids = order_book['bids']
    asks = order_book['asks']

    bid_price_differences = [bids[i][0] - bids[i+1][0]
                             for i in range(len(bids)-1)]
    ask_price_differences = [asks[i+1][0] - asks[i][0]
                             for i in range(len(asks)-1)]

    min_bid_tick_size = min(bid_price_differences)
    min_ask_tick_size = min(ask_price_differences)

    value_difference = abs(num1 - num2)

    tick_size = min(min_bid_tick_size, min_ask_tick_size)

    num_pips = round(value_difference / tick_size)
    return num_pips


def count_digits_in_float(num):
    if isinstance(num, float):
        num_str = str(num)
        integer_part, _, fractional_part = num_str.partition('.')
        return len(integer_part) + len(fractional_part)
    else:
        return 0


# def add_deviation(number, deviation):
#     num_str = str(number)
#     integer_part, decimal_part = num_str.split('.')
#     last_six_digits = decimal_part[-6:]
#     last_six_digits_int = int(last_six_digits)
#     new_last_six_digits = 0
#     if is_minus == False:
#         new_last_six_digits = str(last_six_digits_int + deviation).zfill(6)
#     if is_minus == True:
#         new_last_six_digits = str(last_six_digits_int - deviation).zfill(6)
#     new_last_six_digits = new_last_six_digits.replace("-", "")
#     new_decimal_part = decimal_part[:-6] + new_last_six_digits
#     new_number_str = integer_part + '.' + new_decimal_part
#     new_number = float(new_number_str)
#     return new_number

def add_deviation(order_book, given_value, is_minus, num_pips):
    # Extract the bid and ask price levels from the order book
    bids = order_book['bids']
    asks = order_book['asks']

    # Calculate the price differences between adjacent bid and ask levels
    bid_price_differences = [bids[i][0] - bids[i+1][0]
                             for i in range(len(bids)-1)]
    ask_price_differences = [asks[i+1][0] - asks[i][0]
                             for i in range(len(asks)-1)]

    # Determine the smallest price difference (tick size)
    min_bid_tick_size = min(bid_price_differences)
    min_ask_tick_size = min(ask_price_differences)

    price_change_up = given_value + (min_ask_tick_size * num_pips)

    price_change_down = given_value - (min_bid_tick_size * num_pips)

    if (is_minus):
        return price_change_down
    else:
        return price_change_up


def addZeroToLastDigit(num: float):
    str_num = str(num) + '0'
    return str_num


def calculate_profit_ratio(open_rate, exit_rate, is_short, leverage, stake_ammount):
    quantity = (stake_ammount*leverage)/open_rate
    initial_margin = quantity * open_rate * (1/leverage)
    pnl = 0
    roi = 0
    if (is_short == False):
        pnl = (exit_rate - open_rate) * quantity
    else:
        pnl = (open_rate - exit_rate) * quantity

    roi = pnl / initial_margin
    return round(roi * 100, 2)


def last_seven_day_in_int():
    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=7)
    days_list = []
    for delta in range((current_date - start_date).days + 1):
        day = start_date + timedelta(days=delta)
        days_list.append(day.day)
    return days_list
