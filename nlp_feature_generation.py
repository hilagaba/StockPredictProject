import csv
import pickle as pkl
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
from feature_generator import *
from config import *
# nltk.download('vader_lexicon')
from bs4 import BeautifulSoup
from urllib.request import urlopen

# In this file we have everything related to nlp

''' 
    :returns - max date in the data 
'''


def get_max_date():
    snp_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return snp_data.Close.index[-1]

'''
    :description - this functions get the news' headlines from the web if it is 
    wanted. Afterwards, create features from it and also get true labels,
    also only if it's wanted.
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode   
    :returns df - data frame that hold all data's features
    :returns true_label - the true label corresponding to each row in df
    (Same, Up, Down labels)
'''


def prepare_nlp_features_reuter(opt):
    get_news() if opt.nlp in ("get_news", "full_flow") else None
    df, true_label = prepare_nlp_features_from_reuters() \
        if opt.nlp in ("generate_features", "full_flow") else get_nlp_features()
    return df, true_label

'''
    :description - this function create statical features for a given field from
    the sentiment analysis output (positive, negative, neutral, compound)
    :returns df - data frame that hold all data's features 
    :param field - will be one of the field given from the sentiment analysis' 
    output (positive, negative, neutral, compound)
'''


def get_features(df, field):
    df['MA_' + field] = df[field].rolling(window=ma).mean()
    df['MA_50_' + field] = df[field].rolling(window=ma_50).mean()
    df['Fast_MA_' + field] = df[field].rolling(window=ma_fast).mean()
    df['Slow_MA_' + field] = df[field].rolling(window=ma_slow).mean()
    df['Ema_19_Weekly_' + field] = ema_fast(df[field], 19 * 5)
    df['Ema_39_Weekly_' + field] = ema_slow(df[field], 39 * 5)
    df['Ema_19_Weekly_ABOVE_Ema_39_W_' + field] = (df['Ema_19_Weekly_' + field] - df['Ema_39_Weekly_' + field]) > 0
    df['EMA_19W_39W_DIFF_' + field] = df['Ema_39_Weekly_' + field] - df['Ema_19_Weekly_' + field]
    df['EMA_19W_39W_CROSS_DOWN_' + field] = get_cross_down(fast_ma=df['Ema_19_Weekly_' + field], slow_ma=df['Ema_39_Weekly_' + field])
    df['EMA_19W_39W_CROSS_UP_' + field] = get_cross_up(fast_ma=df['Ema_19_Weekly_' + field], slow_ma=df['Ema_39_Weekly_' + field])
    df['CROSS_DOWN_' + field] = get_cross_down(fast_ma=df['Fast_MA_' + field], slow_ma=df['Slow_MA_' + field])
    df['CROSS_UP_' + field] = get_cross_up(fast_ma=df['Fast_MA_' + field], slow_ma=df['Slow_MA_' + field])
    df['MA_DIFF_' + field] = df['Fast_MA_' + field] - df['Slow_MA_' + field]
    df['STD_' + field] = df[field].rolling(window=ma).std()
    df['Upper_Band_' + field] = df['MA_' + field] + (df['STD_' + field] * 2)
    df['Lower_Band_' + field] = df['MA_' + field] - (df['STD_' + field] * 2)
    df['Slow_Ema_' + field] = ema_slow(df[field], 8)
    df['Fast_Ema_' + field] = ema_fast(df[field], 5)

'''
    :description - get titles for articles from reuters website, according max
    date in historical data, and starting 2007-01-01 (which is the first date 
    available in the website in the time we've work on the project). 
'''


def get_news():
    max_date = get_max_date()
    years = [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
    years.reverse()
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', 10, 11, 12]
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    base_url = 'https://www.reuters.com/resources/archive/us/'
    suffix_url = '.html'
    if not os.path.exists(os.path.join(os.getcwd(), "NLP", "data")):
        os.makedirs(os.path.join(os.getcwd(), "NLP", "data"), exist_ok=True)
    with open(news_csv_path, 'w', newline='', encoding='utf-8') as news_csv_file:
        flag = True
        news_csv_writer = csv.writer(news_csv_file, dialect='excel')
        for year in years:
            for month in months:
                for day in days:
                    try:
                        datetime(int(year), int(month), int(day))
                    except ValueError:
                        continue
                    if datetime(int(year), int(month), int(day)) > max_date:
                        break
                    print('Article date: {}-{}-{}, time {}'.format(year, month, day, str(datetime.now())))
                    article_date = '{}-{}-{}'.format(year, month, day)
                    daily_titles = get_titles('{}{}{}{}{}'.format(base_url, year, month, day, suffix_url))

                    if daily_titles is None:
                        continue
                    if flag:
                        news_csv_writer.writerow(["Date", "Titles"])
                        flag = False
                    news_csv_writer.writerow([article_date, daily_titles])

'''
    :description - this function gets the titles from a daily web page.
    :param daily_url - a url for a give day.
    :returns - titles from a daily web page.
'''


def get_titles(daily_url):
    for i in range(0, 3):
        try:
            article_data = urlopen(daily_url).read()
            break
        except:
            if i == 2:
                print("Failed to open link: {}".format(daily_url))
                return None
            else:
                time.sleep(2)
    soup = BeautifulSoup(article_data, features="lxml")
    content = soup.find_all("div",   attrs={"class": "headlineMed"})
    titles = ""
    for i, _ in enumerate(content):
        url = content[i].a["href"]
        if not ("/news/picture/" in url) or not ("/news/video/" in url) or not ("article/pictures-report" in url) \
                or not ("/article/life-" in url):
            titles += str(content[i].a.contents[0]) + ". "  # .encode('utf-8')
    return titles.rstrip().replace(",", " ").replace("\n", "n")

'''
    :description - gets news' headlines and true labels. Operate sentiment 
    analysis for the headlines and create more statical features in the data frame.
    :returns df - data frame that hold all data's features
    :returns true_label - the true label corresponding to each row in df
    (Same, Up, Down labels)  
'''


def prepare_nlp_features_from_reuters():
    news = pd.read_csv(news_csv_path, index_col=0, parse_dates=True)
    snp_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = []
    for date in news.index:
        if date in snp_data.Close:
            prices.append(snp_data.Close[date])
        else:
            prices.append(math.nan)
    news['Close'] = prices
    news = news[['Close', 'Titles']]
    df = news[['Close']].copy()
    sid = SentimentIntensityAnalyzer()
    for date, row in news.T.iteritems():
        try:
            sentence = unicodedata.normalize('NFKD', news.loc[date, 'Titles']).encode('ascii', 'ignore')
            ss = sid.polarity_scores(str(sentence))
            df.at[date, 'compound'] = ss['compound']
            df.at[date, 'neg'] = ss['neg']
            df.at[date, 'neu'] = ss['neu']
            df.at[date, 'pos'] = ss['pos']
        except TypeError:
            continue
    for field in ['pos', 'neu', 'neg', 'compound']:
        get_features(df, field)
    df = df.dropna()
    true_label = get_label_from_data(df['Close'])
    true_label = true_label[-1 * len(df['Close']):]  # fix the shift from the features after removing Nan \ empty values
    if not os.path.exists(os.path.join(os.getcwd(), "NLP", "data")):
        os.makedirs(os.path.join(os.getcwd(), "NLP", "data"), exist_ok=True)
    df.to_csv(feature_reuters_csv_path)
    return df, true_label

'''
    :description - get features and true labels from file
    :returns features - the nlp features
    :returns true_label - true labels (Same, Up, Down)
'''


def get_nlp_features():
    features = pd.read_csv(feature_reuters_csv_path, index_col=0, parse_dates=True)
    features = features.dropna()
    true_label = get_label_from_data(features['Close'])
    true_label = true_label[-1 * len(features['Close']):]  # fix the shift from the features after removing Nan \ empty values
    return features, true_label