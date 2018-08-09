import csv
from config import *
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
from stockstats import StockDataFrame
from ta import *
from scipy.interpolate import UnivariateSpline


def csv_to_pd():
    feature_csv_path = os.path.join(os.path.dirname(csv_path), "features.csv")
    with open(csv_path) as csv_data_file:
        print("Reading csv file:\n \t\t\t\t{}\n".format(csv_path))
        csv_reader = csv.reader(csv_data_file)
        next(csv_reader, None)
        with open(feature_csv_path, 'w', newline='') as feature_file:
            csvwriter = csv.writer(feature_file)
            csvwriter.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'])
            for row in csv_reader:
                # Date, Open, High, Low, Close, Adj Close, Volume
                stock_date = row[0]
                stock_open = row[1]
                stock_high = row[2]
                stock_low = row[3]
                stock_close = row[4]
                stock_adj_close = row[5]
                stock_volume = row[6]
                csvwriter.writerow([stock_date, stock_open, stock_high, stock_low, stock_close,
                                    stock_adj_close, stock_volume])
    df = pd.read_csv(feature_csv_path, index_col=0, parse_dates=True)
    df['MIDDLE'] = (df.High + df.Low + df.Close) / 3
    df['HIGH-LOW'] = df.High - df.Low
    df['CLOSE-OPEN'] = df.Close - df.Open
    df['OVER_2DAYS'] = df.Open / df.Close.shift(-1) - 1
    df['OVER_NIGHT'] = df.Open.shift(-1) / df.Close - 1
    df['DERIVATIVE'] = get_derivative(df.Low, df.High)
    df['MA'] = df['Adj_Close'].rolling(window=ma).mean()
    df['MA_50'] = df['Adj_Close'].rolling(window=ma_50).mean()
    df['MA_200'] = df['Adj_Close'].rolling(window=ma_200).mean()
    df['Fast_MA'] = df['Adj_Close'].rolling(window=ma_fast).mean()
    df['Slow_MA'] = df['Adj_Close'].rolling(window=ma_slow).mean()
    df['Ema_19_Weekly'] = ema_fast(df.Close, 19 * 5)
    df['Ema_39_Weekly'] = ema_slow(df.Close, 39 * 5)

    df['Ema_19_Weekly_ABOVE_Ema_39_W'] = (df['Ema_19_Weekly'] - df['Ema_39_Weekly']) > 0
    df['PRICE_ABOVE_Ema_19W'] = (df.Close - df['Ema_19_Weekly']) > 0
    df['PRICE_ABOVE_Ema_39w'] = (df.Close - df['Ema_39_Weekly']) > 0
    df['PRICE_EMA19W_DIFF'] = df.Close - df['Ema_19_Weekly']
    df['PRICE_EMA39_DIFF'] = df.Close - df['Ema_39_Weekly']
    df['EMA_19W_39W_DIFF'] = df['Ema_39_Weekly'] - df['Ema_19_Weekly']
    df['EMA_19W_39W_CROSS_DOWN'] = get_cross_down(fast_ma=df['Ema_19_Weekly'], slow_ma=df['Ema_39_Weekly'])
    df['EMA_19W_39W_CROSS_UP'] = get_cross_up(fast_ma=df['Ema_19_Weekly'], slow_ma=df['Ema_39_Weekly'])

    df['CROSS_DOWN'] = get_cross_down(fast_ma=df['Fast_MA'], slow_ma=df['Slow_MA'])
    df['CROSS_UP'] = get_cross_up(fast_ma=df['Fast_MA'], slow_ma=df['Slow_MA'])
    df['FAST_ABOVE_SLOW'] = (df['Fast_MA'] - df['Slow_MA']) > 0
    df['PRICE_ABOVE_MA50'] = (df.Close - df['MA_50']) > 0
    df['PRICE_ABOVE_MA200'] = (df.Close - df['MA_200']) > 0
    df['PRICE_ABOVE_MA'] = (df.Close - df['MA']) > 0
    df['PRICE_ABOVE_MA_FAST'] = (df.Close - df['Fast_MA']) > 0
    df['PRICE_ABOVE_MA_SLOW'] = (df.Close - df['Slow_MA']) > 0
    df['PRICE_MA50_DIFF'] = df.Close - df['MA_50']
    df['PRICE_MA200_DIFF'] = df.Close - df['MA']
    df['PRICE_MA_SLOW_DIFF'] = df.Close - df['Slow_MA']
    df['PRICE_MA_FAST_DIFF'] = df.Close - df['Fast_MA']
    df['PRICE_MA_DIFF'] = df.Close - df['MA_200']
    df['MA_DIFF'] = df['Fast_MA'] - df['Slow_MA']
    df['MA_50-200_DIFF'] = df['MA_200'] - df['MA_50']
    df['STD'] = df['Adj_Close'].rolling(window=ma).std()
    df['Upper_Band'] = df['MA'] + (df['STD'] * 2)
    df['Lower_Band'] = df['MA'] - (df['STD'] * 2)
    df['ADX'] = adx(df.High, df.Low, df.Close)
    df['CCI'] = cci(df.High, df.Low, df.Close)
    df['Slow_Ema'] = ema_slow(df.Close, 13)
    df['Fast_Ema'] = ema_fast(df.Close, 5)
    df['Fast_Ema'] = ema_slow(df.Close, 39)
    df['MACD'] = macd(df.Close)

    df['RSI'] = rsi(df.Close)
    df['RSI_CROSS_30_UP'] = get_indicator_cross_up_known_value(df.RSI, 30)
    df['RSI_CROSS_30_DOWN'] = get_indicator_cross_down_known_value(df.RSI, 30)
    df['RSI_CROSS_70_DOWN'] = get_indicator_cross_down_known_value(df.RSI, 70)
    df['RSI_CROSS_70_UP'] = get_indicator_cross_up_known_value(df.RSI, 70)
    df['RSI_BIGGER_THAN_50'] = df['RSI'] > 50

    df['STOCH'] = stoch(df.High, df.Low, df.Close)
    df['ABSOLUTE_CHANGE'] = get_absolute_change(df.Close)
    df['SEQUENCE_INDEX'] = get_serial_num_in_sequence(get_absolute_change(df.Close))
    df['CHANGE'] = get_change_price(df['Close'])
    # x = [x for x,_ in enumerate(df.Close)]
    # y_spl = UnivariateSpline(enumerate(df.Close), df.Close, s=0, k=4)
    # y_spl_1d = y_spl.derivative(n=1)
    true_label = get_label_from_data(df['Close'])
    df = df.dropna()
    true_label = true_label[-1 * len(df.Close):]  # fix the shift from the features after removing Nan \ empty values
    return df, true_label


def get_change_price(closing_prices):
    change_prices = [100 * (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
                     for i in range(1, len(closing_prices))]
    change_prices.insert(0, 0.0)  # Insert 0 change to the first element
    return change_prices


def get_label_from_data(closing_prices):
    labels = 100 * (closing_prices.shift(-1) - closing_prices) / closing_prices
    if len(classes) == 3:
        true_label = ["Up" if x >= threshold else "Down" if x <= -1 * threshold else "Same" for x in labels]
    else:
        true_label = ["Up" if x >= threshold else "Down" for x in labels]
        # true_label = np.where(df['Close'].shift(-1) > df['Close'], "Up", "Down")
    return true_label


def get_absolute_change(closing_prices):
    change_prices = [(closing_prices[i] - closing_prices[i - 1]) for i in range(1, len(closing_prices))]
    change_prices.insert(0, 0.0)  # Insert 0 change to the first element
    return change_prices


def get_serial_num_in_sequence(change_prices):
    indices_seq = []
    # indices_seq = [1 if i == 0 else indices_seq[i-1]+1 if np.sign(change_prices[i]) == np.sign(change_prices[i-1])
    #                else 1 for i in range(0, len(change_prices))]

    for i in range(0, len(change_prices)):
        if i == 0:
            indices_seq.append(1)
        elif np.sign(change_prices[i]) == np.sign(change_prices[i - 1]):
            indices_seq.append(indices_seq[i - 1] + 1)
        else:
            indices_seq.append(1)
    return indices_seq


def get_derivative(low_prices, high_prices):
    up = [0 if i < ws else (high_prices[i] - low_prices[i - ws]) / ws for i in range(0, len(high_prices))]
    down = [0 if i < ws else (low_prices[i] - high_prices[i - ws]) / ws for i in range(0, len(high_prices))]
    return [i if abs(i) > abs(j) else j for i, j in zip(up, down)]


def get_cross_down(fast_ma, slow_ma):
    up = (fast_ma.shift(-1) - slow_ma.shift(-1)) > 0
    down = (fast_ma - slow_ma) <= 0
    return [x and y for x, y in zip(up, down)]


def get_cross_up(fast_ma, slow_ma):
    up = (fast_ma.shift(-1) - slow_ma.shift(-1)) < 0
    down = (fast_ma - slow_ma) >= 0
    return [x and y for x, y in zip(up, down)]


def get_indicator_cross_down_known_value(indicator, value):
    indicator_list = [indicator[i-1] >= value and indicator[i] < value for i in range(1, len(indicator))]
    while len(indicator_list) < len(indicator):
        indicator_list.append(False)
    return indicator_list


def get_indicator_cross_up_known_value(indicator, value):
    indicator_list = [indicator[i - 1] <= value and indicator[i] > value for i in range(1, len(indicator))]
    while len(indicator_list) < len(indicator):
        indicator_list.append(False)
    return indicator_list
