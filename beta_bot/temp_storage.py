from enum import Enum


class TempStorage(str, Enum):
    command_for_run = "command_for_run"
    config = "config"
    cursor = "cursor"
    conn = "conn"
    exchange = "exchange"
    telegramBot = "telegramBot"
    botStartDate = ""
    dp = ""
    future_prediction = "future_prediction"
    dataframe = "dataframe"
    should_count_for_current = "should_count_for_current"
    current_data = "current_data"
    conditionToAddNew = "conditionToAddNew"


temp_storage_data = {
    TempStorage.command_for_run: "start",  # start stop
    TempStorage.future_prediction: {},
    TempStorage.dataframe: {},
    TempStorage.should_count_for_current: {},
    TempStorage.current_data: {},
    TempStorage.conditionToAddNew: {},
}
