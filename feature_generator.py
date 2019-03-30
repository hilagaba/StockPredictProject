from config import *
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from stockstats import StockDataFrame
from ta import *

# In this file the features are being created according the data and also
# true labels.


''' 
    :description - features are created to the historical data mode
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns df - data frame that hold all data's features
    :returns true_label - the true label corresponding to each row in df
'''


def csv_to_pd(opt):
    print("Calculating features, time {}".format(str(datetime.now())))
    logger.info("Calculating features, time {}".format(str(datetime.now())))
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df['MVAR'], df['MSTD'] = get_mvar_mstd(df['Low'], df['High'], df['Close'], df['Open'], df['Volume'])
    df['WR'] = wr(df['High'], df['Low'], df['Close'])
    df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'])
    df['MFI'] = money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    df["UO"] = uo(df['High'], df['Low'], df['Close'])
    df["stoch_signal"] = stoch_signal(df['High'], df['Low'], df['Close'])
    df["AO"] = ao(df['High'], df['Low'])
    df["macd_signal"] = macd_signal(df['Close'])
    df["macd_diff"] = macd_diff(df['Close'])
    df["adx_pos"] = adx_pos(df['High'], df['Low'], df['Close'])
    df["adx_neg"] = adx_neg(df['High'], df['Low'], df['Close'])
    df["adx_indicator"] = adx_indicator(df['High'], df['Low'], df['Close'])
    df["vortex_indicator_pos"] = vortex_indicator_pos(df['High'], df['Low'], df['Close'])
    df["vortex_indicator_neg"] = vortex_indicator_neg(df['High'], df['Low'], df['Close'])
    df["TRIX"] = trix(df['Close'])
    df["mass_index"] = mass_index(df['High'], df['Low'])
    df["ICHIMOKU"] = ichimoku_a(df['High'], df['Low'])
    df["keltner_channel_central"] = keltner_channel_central(df['High'], df['Low'], df['Close'])
    df["keltner_channel_lband"] = keltner_channel_lband(df['High'], df['Low'], df['Close'])
    df["keltner_channel_hband"] = keltner_channel_hband(df['High'], df['Low'], df['Close'])
    df["VPT"] = volume_price_trend(df['Close'], df['Volume'])
    df["FI"] = force_index(df['Close'], df['Volume'])
    df["CMF"] = chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
    df["OBV"] = on_balance_volume(df['Close'], df['Volume'])
    df["ADI"] = acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

    df['MIDDLE'] = (df.High + df.Low + df.Close) / 3
    df['HIGH-LOW'] = df.High - df.Low
    df['CLOSE-OPEN'] = df.Close - df.Open
    df['OVER_2DAYS'] = df.Open / df.Close.shift(-1) - 1
    df['OVER_NIGHT'] = df.Open.shift(-1) / df.Close - 1
    df['DERIVATIVE'] = get_derivative(df.Low, df.High)
    df['2ND_DERIVATIVE'] = get_derivative(df['DERIVATIVE'], df['DERIVATIVE'])
    df['MA'] = df['Adj Close'].rolling(window=ma).mean()
    df['MA_50'] = df['Adj Close'].rolling(window=ma_50).mean()
    df['MA_200'] = df['Adj Close'].rolling(window=ma_200).mean()
    df['Fast_MA'] = df['Adj Close'].rolling(window=ma_fast).mean()
    df['Slow_MA'] = df['Adj Close'].rolling(window=ma_slow).mean()
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
    df['PRICE_MA200_DIFF'] = df.Close - df['MA_200']
    df['PRICE_MA_SLOW_DIFF'] = df.Close - df['Slow_MA']
    df['PRICE_MA_FAST_DIFF'] = df.Close - df['Fast_MA']
    df['PRICE_MA_DIFF'] = df.Close - df['MA']
    df['MA_DIFF'] = df['Fast_MA'] - df['Slow_MA']
    df['MA_50-200_DIFF'] = df['MA_200'] - df['MA_50']
    df['STD'] = df['Adj Close'].rolling(window=ma).std()
    df['Upper_Band'] = df['MA'] + (df['STD'] * 2)
    df['Lower_Band'] = df['MA'] - (df['STD'] * 2)
    df['ADX'] = adx(df.High, df.Low, df.Close)
    df['CCI'] = cci(df.High, df.Low, df.Close)
    df['Slow_Ema'] = ema_slow(df.Close, 8)
    df['Fast_Ema'] = ema_fast(df.Close, 5)
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
    true_label = get_label_from_data(df['Close'])

    if opt.enable_vix:
        df_VIX = pd.read_csv(os.path.join(os.path.dirname(csv_path), "^VIX.csv"), index_col=0, parse_dates=True)
        df['Close_VIX'] = df_VIX['Close']
        df['Open_VIX'] = df_VIX['Open']
        df['High_VIX'] = df_VIX['High']
        df['Low_VIX'] = df_VIX['Low']

    df = df.dropna()
    true_label = true_label[-1 * len(df.Close):]  # fix the shift from the features after removing Nan \ empty values
    true_label = true_label[-1 * len(df.Close):]  # fix the shift from the features after removing Nan \ empty values

    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.mkdir(os.path.join(os.getcwd(), "data"))
    feature_csv_path = os.path.join(os.path.dirname(csv_path), "features.csv")
    df.to_csv(feature_csv_path)
    return df, true_label


'''
    :description - the function calculate the close price diff between each 2
    following days
    :param closing_prices - all data's close prices 
    :returns - the close data diff for each day from its previous one (in percents)
'''


def get_change_price(closing_prices):
    change_prices = [100 * (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
                     for i in range(1, len(closing_prices))]
    change_prices.insert(0, 0.0)  # Insert 0 change to the first element
    return change_prices

'''
    :description - the function calculate the true label by calculating the diff
    between close price of tomorrow to today
    :param closing_prices - all data's close prices 
    :returns - label for each day (Up, Same, Down)
'''


def get_label_from_data(closing_prices):
    change = 100 * (closing_prices.shift(-1) - closing_prices) / closing_prices
    if len(classes) == 3:
        true_label = ["Up" if x >= threshold else "Down" if x <= -1 * threshold else "Same" for x in change]
    else:
        true_label = ["Up" if x >= threshold else "Down" for x in change]
    return true_label


'''
    :description - the function calculate the close price diff between each 2
    following days
    :param closing_prices - all data's close prices 
    :returns - the close data diff for each day from its previous one
'''


def get_absolute_change(closing_prices):
    change_prices = [(closing_prices[i] - closing_prices[i - 1]) for i in range(1, len(closing_prices))]
    change_prices.insert(0, 0.0)  # Insert 0 change to the first element
    return change_prices

'''
    :description - this function calculate the sequence number of a day
    according to some continuous trend
    :param change_prices - all data's change prices, meaning diff in close price
    from today to yesterday
    :returns indices_seq - returns the feature's values
'''


def get_serial_num_in_sequence(change_prices):
    indices_seq = []
    for i in range(0, len(change_prices)):
        if i == 0:
            indices_seq.append(1)
        elif np.sign(change_prices[i]) == np.sign(change_prices[i - 1]):
            indices_seq.append(indices_seq[i - 1] + 1)
        else:
            indices_seq.append(1)
    return indices_seq

'''
    :description - calculate the derivative for a period of 5 days
    :param low_prices - the lowest price for each day
    :param high_prices - the highest price for each day
    :returns - the features' values
'''


def get_derivative(low_prices, high_prices):
    up = [0 if i < ws else (high_prices[i] - low_prices[i - ws]) / ws for i in range(0, len(high_prices))]
    down = [0 if i < ws else (low_prices[i] - high_prices[i - ws]) / ws for i in range(0, len(high_prices))]
    return [i if abs(i) > abs(j) else j for i, j in zip(up, down)]


'''
    :description - calculate feature cross down. It's a binary feature,
    the indicate if the moving average in the last days is bigger than 
    the moving average on a longer period of the last days. 
    :param fast_ma - the moving average on the shorter period of the last days
    :param slow_ma - the moving average on the longer period of the last days 
    :returns - the features' values
'''


def get_cross_down(fast_ma, slow_ma):
    up = (fast_ma.shift(-1) - slow_ma.shift(-1)) > 0
    down = (fast_ma - slow_ma) <= 0
    return [x and y for x, y in zip(up, down)]


'''
    :description - calculate feature cross up. It's a binary feature,
    the indicate if the moving average in the last days is smaller than 
    the moving average on a longer period of the last days. 
    :param fast_ma - the moving average on the shorter period of the last days
    :param slow_ma - the moving average on the longer period of the last days 
    :returns - the features' values
'''


def get_cross_up(fast_ma, slow_ma):
    up = (fast_ma.shift(-1) - slow_ma.shift(-1)) < 0
    down = (fast_ma - slow_ma) >= 0
    return [x and y for x, y in zip(up, down)]

'''
    :description - calculate the feature RSI_CROSS_X_DOWN
    :param indicator - rsi features' values
    :param value - some threshold
    :returns - the feature's values
'''


def get_indicator_cross_down_known_value(indicator, value):
    indicator_list = [indicator[i-1] >= value and indicator[i] < value for i in range(1, len(indicator))]
    while len(indicator_list) < len(indicator):
        indicator_list.append(False)
    return indicator_list


'''
    :description - calculate the feature RSI_CROSS_X_UP
    :param indicator - rsi features' values
    :param value - some threshold
    :returns - the feature's values
'''


def get_indicator_cross_up_known_value(indicator, value):
    indicator_list = [indicator[i - 1] <= value and indicator[i] > value for i in range(1, len(indicator))]
    while len(indicator_list) < len(indicator):
        indicator_list.append(False)
    return indicator_list


'''
    :description - calculate 2 features: MVAR, MSTD
    :param low - low prices features 
    :param high - high prices features
    :param close - close prices features
    :param open - open prices features
    :param volume - volume features
    :returns close_14_mvar feature and close_14_mstd feature
'''


def get_mvar_mstd(low, high, close, open, volume):
    df_for_stockstat = pd.DataFrame(columns=['low', 'high', 'close', 'open', 'volume'])
    df_for_stockstat['low'] = low
    df_for_stockstat['high'] = high
    df_for_stockstat['close'] = close
    df_for_stockstat['open'] = open
    df_for_stockstat['volume'] = volume
    stockstat = StockDataFrame(df_for_stockstat)
    stockstat._get_mvar(df_for_stockstat, 'close', 14)
    stockstat._get_mstd(df_for_stockstat, 'close', 14)
    return df_for_stockstat['close_14_mvar'], df_for_stockstat['close_14_mstd']


