import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from config import *
import seaborn as sns
from win32api import GetSystemMetrics
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter


'''
    :description - this function responsible for all data visualization  
    :param df - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in df 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode 
'''


def visualization(df, true_label, opt):
    snp500_chart(df) if not opt.nlp else None
    observations_per_class(true_label, opt)
    correlation_feature_vs_class(df, true_label, opt)
    percent_label_per_month(df, true_label) if not opt.nlp else None
    feature_t_test_correlation(df, true_label, opt)
    normalized_feature_correlation(df, true_label, opt)
    feature_correlation(df, true_label, opt)

'''
    :description - plot snp500 charts
    :param df - data frame that hold all data's features 
'''


def snp500_chart(df):
    print("Plot S&P figure, time {}".format(str(datetime.now())))
    logger.info("Plot S&P figure, time {}".format(str(datetime.now())))
    plt.figure()
    df.dropna()
    plt.plot(df.Close, label="close")
    plt.title("S&P 500 close price chart")
    plt.ylabel("S&P500 Price")
    plt.xlabel("Date")
    plt.legend(loc='best')
    plt.show(block=False)
    if not os.path.exists(os.path.join(os.getcwd(), "data_analysis")):
        os.mkdir(os.path.join(os.getcwd(), "data_analysis"))
    plt.savefig(os.path.join(os.path.join(os.getcwd(), "data_analysis"), "S&P500_close_price.png"))
    # plot with volume
    plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax1.plot(df.Close, label="close")
    plt.title("S&P 500 close price chart")
    plt.ylabel("S&P500 Price")
    plt.legend(loc='best')
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    plt.ylabel("Volume")
    plt.xlabel("Date")
    ax2.bar(df.index, df.Volume)
    plt.show(block=False)
    if not os.path.exists(os.path.join(os.getcwd(), "data_analysis")):
        os.mkdir(os.path.join(os.getcwd(), "data_analysis"))
    plt.savefig(os.path.join(os.path.join(os.getcwd(), "data_analysis"), "S&P500_close_price_volume.png"))
    # plot candlestick ohlc with volume
    # Get width & height resolution
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    fig = plt.figure(figsize=(width / 100., height / 100.), dpi=400)
    ohcl = []
    for i, _ in enumerate(df.index):
        # OHLC support date2num as xaxix
        args = mdates.date2num(df.index[i]), df.Open[i], df.High[i], df.Low[i], df.Close[i]
        ohcl.append(args)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax1.xaxis_date()
    plt.tight_layout()
    candlestick_ohlc(ax1, ohcl, width=0.4, colordown="#db3f3f", colorup="#77d879")
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax2.fill_between(df.Volume.index.map(mdates.date2num), df.Volume.values, 0)
    plt.ylabel("Volume")
    plt.xlabel("Date")
    ax2.bar(df.index, df.Volume)
    plt.subplots_adjust(left=0.09, bottom=0.2, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.tight_layout()
    plt.show(block=False)
    if not os.path.exists(os.path.join(os.getcwd(), "data_analysis")):
        os.mkdir(os.path.join(os.getcwd(), "data_analysis"))
    plt.savefig(os.path.join(os.path.join(os.getcwd(), "data_analysis"), "S&P500_close_price_candlestick_volume.png"))


'''
    :description - this function plot the number of observation for each class
    in columns graph 
    :param true_labels - the true labels of all data
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    
'''


def observations_per_class(true_labels, opt):
    print("Plot observation per class, time {}".format(str(datetime.now())))
    logger.info("Plot observation per class, time {}".format(str(datetime.now())))
    y_pos = np.arange(len(classes))
    quantities = [true_labels.count('Up'), true_labels.count('Same'), true_labels.count('Down')]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar = plt.bar(y_pos, quantities, align='center', alpha=0.5)
    for i, j in zip(y_pos, quantities):
        ax.annotate(str(j), xy=(i, j))

    bar[0].set_color('g')
    bar[1].set_color('black')
    bar[2].set_color('r')
    plt.xticks(y_pos, classes)
    plt.ylabel('Quantities of observations')
    plt.xlabel('Classes')
    plt.title('Number of observations per classes ({} same threshold)'.format(threshold))
    plt.show(block=False)
    path = os.path.join(os.getcwd(), "data_analysis") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data_analysis")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "Number of observations per classes.png"))
    quantities_prc = [100 * true_labels.count('Up') / len(true_labels), 100 * true_labels.count('Same') /
                      len(true_labels), 100 * true_labels.count('Down') / len(true_labels)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar = plt.bar(y_pos, quantities_prc, align='center', alpha=0.5)
    for i, j in zip(y_pos, quantities_prc):
        k = int(j)
        ax.annotate(str(k), xy=(i, k))
    bar[0].set_color('g')
    bar[1].set_color('black')
    bar[2].set_color('r')
    plt.xticks(y_pos, classes)
    plt.ylabel('Percent of observations')
    plt.xlabel('Classes')
    plt.title('Percent of observations per classes ({} same threshold)'.format(threshold))
    plt.show(block=False)
    path = os.path.join(os.getcwd(), "data_analysis") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data_analysis")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "Percent of observations per classes.png"))

'''
    description - this function plot the relation between feature's values
    range and the different classification. Stack columns chart will be created to
    binary features and box plot chart for continuous features.  
    :param df - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in df 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
'''


def correlation_feature_vs_class(df, true_labels, opt):
    print("Plot correlation feature vs class figure, time {}".format(str(datetime.now())))
    logger.info("Plot correlation feature vs class figure, time {}".format(str(datetime.now())))
    # Multiple box plots on one Axes
    for feature_name, feature_data in df.items():
        plt.figure()
        if str(feature_data[0]) in ("True", "False"):  # binary feature
            true = [0, 0, 0]
            false = [0, 0, 0]
            max_labels = max([true_labels.count('Up'), true_labels.count('Same'), true_labels.count('Down')])
            for feature_val, true_label in zip(feature_data, true_labels):
                for idx, label in enumerate(classes):
                    if true_label == label:
                        true[idx] += 1 if feature_val else 0
                        false[idx] += 1 if not feature_val else 0
            # Converting to percents
            for i in range(0, len(true)):
                true[i] = 100 * true[i] / (true[i] + false[i])
                false[i] = 100 - true[i]
            ind = np.arange(len(classes))
            width = 0.35
            p1 = plt.bar(ind, true, width, color='green')
            p2 = plt.bar(ind, false, width, bottom=true, color='red')
            plt.ylabel('Percent')
            plt.title('Percent by class and binary value')
            plt.legend((p1[0], p2[0]), ('True', 'False'))
        else:  # continuous feature
            data = [[], [], []]
            for feature_val, true_label in zip(feature_data, true_labels):
                for idx, class_label in enumerate(classes):
                    if true_label == class_label:
                        data[idx].insert(len(data[idx]), feature_val)
            fig, ax = plt.subplots()
            ax.boxplot(data, sym='')
            plt.ylabel('Scope of values feature')
            plt.xlabel('Classes')
            plt.title("Class correlation by feature: {}".format(feature_name))
            ax.set_xticklabels(['Up', 'Same', 'Down'])
        plt.show(block=False)
        path = os.path.join(os.getcwd(), "correlation_feature_vs_class") \
            if not opt.nlp else os.path.join(os.getcwd(), "NLP", "correlation_feature_vs_class")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "{}.png".format(feature_name)))

'''
    :description - plot the each label percent by month 
    :param date_to_down_percents - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in date_to_down_percents
'''


def percent_label_per_month(date_to_down_percents, true_labels):
    print("Plot percent labels per month, time {}".format(str(datetime.now())))
    logger.info("Plot percent labels per month, time {}".format(str(datetime.now())))
    month_year_up, month_year_same, month_year_down, month_year_all = dict(), dict(), dict(), dict()
    for date, true_label in zip(date_to_down_percents.index, true_labels):
        month_year = str(date.month) + str(date.year)
        if month_year not in month_year_all.keys():
            month_year_all[month_year], month_year_up[month_year], month_year_same[month_year], \
            month_year_down[month_year] = 0, 0, 0, 0
        if true_label == "Up" and month_year in month_year_up:
                month_year_up[month_year] += 1
        elif true_label == "Same" and month_year in month_year_same:
                month_year_same[month_year] += 1
        elif true_label == "Down" and month_year in month_year_down:
                month_year_down[month_year] += 1
        month_year_all[month_year] += 1
    percents = [[], [], []]
    for key in month_year_up:
        percents[0].insert(len(percents[0]), 100 * month_year_up[key] / month_year_all[key])
    for key in month_year_same:
        percents[1].insert(len(percents[1]), 100 * month_year_same[key] / month_year_all[key])
    for key in month_year_down:
        percents[2].insert(len(percents[2]), 100 * month_year_down[key] / month_year_all[key])

    years = [x[-4:] for x in month_year_down.keys()]  # Year is 4 last digits
    months = [x[:-4] for x in month_year_down.keys()]  # Month is 1 or 2 first digits
    dates = []
    for idx, date in enumerate(month_year_up.keys()):
        dates.append(datetime.strptime(str(years[idx]) + "-" + str(months[idx]) + "-1", '%Y-%m-%d').strftime('%m-%d-%Y'))
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('Up Label percent by month and year')
    plt.plot(dates, percents[0], color='green')
    plt.ylabel('Up Percent')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.subplot(3, 1, 2)
    plt.title('Same Label percent by month and year')
    plt.plot(dates, percents[1], color='black')
    plt.ylabel('Same Percent')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax = fig.add_subplot(3, 1, 3)
    plt.title('Down Label percent by month and year')
    plt.xlabel('Dates')
    plt.ylabel('Down Percent')
    years = [x[-4:] for x in month_year_down.keys()]
    plt.plot(dates, percents[2], color='red')
    frequency = 12 * 5  # Show every 5 years (every year appears 12 times in dates and years)
    plt.xticks(dates[::frequency], years[::frequency])
    plt.show(block=False)
    if not os.path.exists(os.path.join(os.getcwd(), "data_analysis")):
        os.mkdir(os.path.join(os.getcwd(), "data_analysis"))
    plt.savefig(os.path.join(os.path.join(os.getcwd(), "data_analysis"), "Label percent by month and year.png"))

'''
    :description - create heat map for the correlation between features.
    After features' values normalization. 
    :param df - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in df 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode 
'''


def normalized_feature_correlation(df, true_label, opt):
    dt = df
    for col in df:
        if str(df[col][0]) in ("True", "False"):  # Binary feature
            binary_to_int = [1 if item is True else -1 for idx, item in enumerate(df[col])]
            dt[col] = binary_to_int
        dt[col] = (dt[col] - dt[col].mean()) / dt[col].std(ddof=0)
    feature_correlation(dt, true_label, opt, "_normalized")


'''
    :description - create heat map for the correlation between features. 
    :param df - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in df 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param normalized - indicate the features' values are normalized and create
    a corresponding file name
'''


def feature_correlation(df, true_label, opt, normalized=""):
    print("Start feature correlation {}".format(str(datetime.now())))
    logger.info("Start feature correlation {}".format(str(datetime.now())))

    # Add true label to the matrix
    df['true_label'] = [-1 if x == "Down" else 0 if x == "Same" else 1 for x in true_label]
    df = df.dropna()

    # calculate the correlation matrix
    corr = df.corr()
    path = os.path.join(os.getcwd(), "data") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_name = "features_correlation{}.csv".format(normalized)
    features_correlation_csv_path = os.path.join(path, file_name)
    corr.to_csv(features_correlation_csv_path)

    # Get width & height resolution
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)

    # plot the heatmap
    fig = plt.figure(figsize=(width / 70., height / 70.), dpi=1500)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1, vmax=1.0, square=True)#, annot=True)
    path = os.path.join(os.getcwd(), "data_analysis") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data_analysis")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.show(block=False)
    plt.savefig(os.path.join(path, "features Heat map{}.pdf".format(normalized)), format="pdf", bbox_inches='tight')

'''
    :description - plots heat map for t test and p_value of features' values vs
    true labels.
    :param df - data frame that hold all data's features
    :param true_label - the true label corresponding to each row in df 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode

'''


def feature_t_test_correlation(df, true_label, opt):
    print("Start feature t test correlation {}".format(str(datetime.now())))
    logger.info("Start feature t test correlation {}".format(str(datetime.now())))

    # Add true label to the matrix
    df['true_label'] = [-1 if x == "Down" else 0 if x == "Same" else 1 for x in true_label]
    df = df.dropna()

    # perform t-test
    tdf = pd.DataFrame()
    p_value_df = pd.DataFrame()
    for col in df:
        if str(df[col][0]) in ("True", "False"):  # Binary feature
            binary_to_int = [1 if item is True else -1 for idx, item in enumerate(df[col])]
            df[col] = binary_to_int
        t_test, p_value = ttest_ind(df[col], df['true_label'], equal_var=False, nan_policy='omit')
        tdf[col] = pd.Series(t_test)
        p_value_df[col] = pd.Series(p_value)

    path = os.path.join(os.getcwd(), "data") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    features_ttest_csv_path = os.path.join(path, "features_ttest.csv")
    features_p_value_csv_path = os.path.join(path, "features_p_value.csv")
    tdf.to_csv(features_ttest_csv_path)
    p_value_df.to_csv(features_p_value_csv_path)

    # Get width & height resolution
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)

    # plot the heatmap t_test
    fig = plt.figure(figsize=(width / 90., height / 50.), dpi=1500)
    sns.heatmap(tdf.transpose(), xticklabels=['true_label'], yticklabels=tdf.columns, linewidths=0.3, vmax=1.0, square=True)
    path = os.path.join(os.getcwd(), "data_analysis") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data_analysis")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.show(block=False)
    plt.savefig(os.path.join(path, "features t-test Heat map.pdf"), format="pdf", bbox_inches='tight')

    # plot the heatmap p_value
    fig = plt.figure(figsize=(width / 120., height / 50.), dpi=200)
    sns.heatmap(p_value_df.transpose(), xticklabels=['true_label'], yticklabels=p_value_df.columns, linewidths=0.3, vmax=1.0,
                square=True)
    path = os.path.join(os.getcwd(), "data_analysis") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "data_analysis")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    plt.show(block=False)
    plt.savefig(os.path.join(path, "features p_value Heat map.pdf"), format="pdf", bbox_inches='tight')


