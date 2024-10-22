import talib.abstract as ta
import numpy as np
import math
import pandas as pd


def adaptiveTrendFinder(df: pd.DataFrame, periods):
    dataframe = df.copy()
    dataframe['trend_direction'] = dataframe.apply(
        (lambda x: calculate_trend_direction(x, dataframe, periods)), axis=1)
    dataframe['trend_direction_temp'] = dataframe['trend_direction'].apply(
        lambda x: x[0])
    dataframe.loc[
        (
            (dataframe['trend_direction_temp'].shift(1) < 0)
            &
            (dataframe['trend_direction_temp'] > 0)
        ),
        'trend_direction_real'] = 1

    dataframe.loc[
        (
            (dataframe['trend_direction_temp'].shift(1) > 0)
            &
            (dataframe['trend_direction_temp'] < 0)
        ),
        'trend_direction_real'] = -1

    return dataframe['trend_direction_temp']


def calculate_trend_direction(x, dataframe, periods):
    # if(x.name == len(dataframe)-1) | (x.name == len(dataframe)-2):
    if (x.name >= periods[len(periods)-1]):
        devMultiplier = 2.0
        # Calculate Deviation,PersionR,Slope,Intercept
        stdDev01, pearsonR01, slope01, intercept01 = calcDev(
            periods[1], dataframe, x.name)
        stdDev02, pearsonR02, slope02, intercept02 = calcDev(
            periods[2], dataframe, x.name)
        # stdDev03, pearsonR03, slope03, intercept03 = calcDev(
        #     periods[3], dataframe, x.name)
        # stdDev04, pearsonR04, slope04, intercept04 = calcDev(
        #     periods[4], dataframe, x.name)
        # stdDev05, pearsonR05, slope05, intercept05 = calcDev(
        #     periods[5], dataframe, x.name)
        # stdDev06, pearsonR06, slope06, intercept06 = calcDev(
        #     periods[6], dataframe, x.name)
        # stdDev07, pearsonR07, slope07, intercept07 = calcDev(
        #     periods[7], dataframe, x.name)
        # stdDev08, pearsonR08, slope08, intercept08 = calcDev(
        #     periods[8], dataframe, x.name)
        # stdDev09, pearsonR09, slope09, intercept09 = calcDev(
        #     periods[9], dataframe, x.name)
        # stdDev10, pearsonR10, slope10, intercept10 = calcDev(
        #     periods[10], dataframe, x.name)
        # stdDev11, pearsonR11, slope11, intercept11 = calcDev(
        #     periods[11], dataframe, x.name)
        # stdDev12, pearsonR12, slope12, intercept12 = calcDev(
        #     periods[12], dataframe, x.name)
        # stdDev13, pearsonR13, slope13, intercept13 = calcDev(
        #     periods[13], dataframe, x.name)
        # stdDev14, pearsonR14, slope14, intercept14 = calcDev(
        #     periods[14], dataframe, x.name)
        # stdDev15, pearsonR15, slope15, intercept15 = calcDev(
        #     periods[15], dataframe, x.name)
        # stdDev16, pearsonR16, slope16, intercept16 = calcDev(
        #     periods[16], dataframe, x.name)
        # stdDev17, pearsonR17, slope17, intercept17 = calcDev(
        #     periods[17], dataframe, x.name)
        # stdDev18, pearsonR18, slope18, intercept18 = calcDev(
        #     periods[18], dataframe, x.name)
        # stdDev19, pearsonR19, slope19, intercept19 = calcDev(
        #     periods[19], dataframe, x.name)

        # Find the highest Pearson's R
        highestPearsonR = max(pearsonR01, pearsonR02,
                              # pearsonR03, pearsonR04, pearsonR05, pearsonR06
                              # pearsonR07, pearsonR08, pearsonR09,
                              # pearsonR10, pearsonR11, pearsonR12, pearsonR13, pearsonR14, pearsonR15, pearsonR16, pearsonR17, pearsonR18, pearsonR19
                              )

        # Determine selected length, slope, intercept, and deviations
        detectedPeriod = 0
        detectedSlope = 0
        detectedIntrcpt = 0
        detectedStdDev = 0

        if highestPearsonR == pearsonR01:
            detectedPeriod = periods[1]
            detectedSlope = slope01
            detectedIntrcpt = intercept01
            detectedStdDev = stdDev01
        elif highestPearsonR == pearsonR02:
            detectedPeriod = periods[2]
            detectedSlope = slope02
            detectedIntrcpt = intercept02
            detectedStdDev = stdDev02
        # elif highestPearsonR == pearsonR03:
        #     detectedPeriod = periods[3]
        #     detectedSlope = slope03
        #     detectedIntrcpt = intercept03
        #     detectedStdDev = stdDev03
        # elif highestPearsonR == pearsonR04:
        #     detectedPeriod = periods[4]
        #     detectedSlope = slope04
        #     detectedIntrcpt = intercept04
        #     detectedStdDev = stdDev04
        # elif highestPearsonR == pearsonR05:
        #     detectedPeriod = periods[5]
        #     detectedSlope = slope05
        #     detectedIntrcpt = intercept05
        #     detectedStdDev = stdDev05
        # elif highestPearsonR == pearsonR06:
        #     detectedPeriod = periods[6]
        #     detectedSlope = slope06
        #     detectedIntrcpt = intercept06
        #     detectedStdDev = stdDev06
        # elif highestPearsonR == pearsonR07:
        #     detectedPeriod = periods[7]
        #     detectedSlope = slope07
        #     detectedIntrcpt = intercept07
        #     detectedStdDev = stdDev07
        # elif highestPearsonR == pearsonR08:
        #     detectedPeriod = periods[8]
        #     detectedSlope = slope08
        #     detectedIntrcpt = intercept08
        #     detectedStdDev = stdDev08
        # elif highestPearsonR == pearsonR09:
        #     detectedPeriod = periods[9]
        #     detectedSlope = slope09
        #     detectedIntrcpt = intercept09
        #     detectedStdDev = stdDev09
        # elif highestPearsonR == pearsonR10:
        #     detectedPeriod = periods[10]
        #     detectedSlope = slope10
        #     detectedIntrcpt = intercept10
        #     detectedStdDev = stdDev10
        # elif highestPearsonR == pearsonR11:
        #     detectedPeriod = periods[11]
        #     detectedSlope = slope11
        #     detectedIntrcpt = intercept11
        #     detectedStdDev = stdDev11
        # elif highestPearsonR == pearsonR12:
        #     detectedPeriod = periods[12]
        #     detectedSlope = slope12
        #     detectedIntrcpt = intercept12
        #     detectedStdDev = stdDev12
        # elif highestPearsonR == pearsonR13:
        #     detectedPeriod = periods[13]
        #     detectedSlope = slope13
        #     detectedIntrcpt = intercept13
        #     detectedStdDev = stdDev13
        # elif highestPearsonR == pearsonR14:
        #     detectedPeriod = periods[14]
        #     detectedSlope = slope14
        #     detectedIntrcpt = intercept14
        #     detectedStdDev = stdDev14
        # elif highestPearsonR == pearsonR15:
        #     detectedPeriod = periods[15]
        #     detectedSlope = slope15
        #     detectedIntrcpt = intercept15
        #     detectedStdDev = stdDev15
        # elif highestPearsonR == pearsonR16:
        #     detectedPeriod = periods[16]
        #     detectedSlope = slope16
        #     detectedIntrcpt = intercept16
        #     detectedStdDev = stdDev16
        # elif highestPearsonR == pearsonR17:
        #     detectedPeriod = periods[17]
        #     detectedSlope = slope17
        #     detectedIntrcpt = intercept17
        #     detectedStdDev = stdDev17
        # elif highestPearsonR == pearsonR18:
        #     detectedPeriod = periods[18]
        #     detectedSlope = slope18
        #     detectedIntrcpt = intercept18
        #     detectedStdDev = stdDev18
        # elif highestPearsonR == pearsonR19:
        #     detectedPeriod = periods[19]
        #     detectedSlope = slope19
        #     detectedIntrcpt = intercept19
        #     detectedStdDev = stdDev19

        else:
            # Default case
            raise Exception(f"Cannot Find Highest PearsonR")

        # Calculate start and end price based on detected slope and intercept
        startPrice = math.exp(
            detectedIntrcpt + detectedSlope * (detectedPeriod - 1))
        endPrice = math.exp(detectedIntrcpt)

        trend_direction = endPrice - startPrice
        return (trend_direction, detectedPeriod, highestPearsonR)
    return (0, 0, 0)


def calcDev(length: int, dataframe: pd.DataFrame, index: int):
    logSource = dataframe['close'].apply(lambda x: math.log(x))
    period_1 = length - 1
    sumX = 0.0
    sumXX = 0.0
    sumYX = 0.0
    sumY = 0.0
    for i in range(1, length+1):
        lSrc = logSource[index+1-i]
        sumX += i
        sumXX += i * i
        sumYX += i * lSrc
        sumY += lSrc

    slope = np.nan_to_num((length * sumYX - sumX * sumY) /
                          (length * sumXX - sumX * sumX))
    average = sumY / length
    intercept = average - (slope * sumX / length) + slope
    sumDev = 0.0
    sumDxx = 0.0
    sumDyy = 0.0
    sumDyx = 0.0
    regres = intercept + slope * period_1 * 0.5
    sumSlp = intercept

    for i in range(1, period_1+1):
        lSrc = logSource[index+1-i]
        dxt = lSrc - average
        dyt = sumSlp - regres
        lSrc -= sumSlp
        sumSlp += slope
        sumDxx += dxt * dxt
        sumDyy += dyt * dyt
        sumDyx += dxt * dyt
        sumDev += lSrc * lSrc

    unStdDev = math.sqrt(sumDev / period_1)
    divisor = sumDxx * sumDyy
    if divisor == 0 or np.isnan(divisor):
        pearsonR = 0  # Set Pearson correlation coefficient to NaN
    else:
        pearsonR = sumDyx / math.sqrt(divisor)
    return unStdDev, pearsonR, slope, intercept
