import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import *


def plot_figure(df):
    plt.figure()
    df.dropna()
    plt.plot(df.Close, label="close")
    plt.title("S&P 500 close price chart")
    plt.ylabel("S&P500 Price")
    plt.xlabel("Date")
    plt.legend(loc='best')
    plt.show(block=False)

    # plot with volume
    # plt.figure()
    # ax1 = plt.sub*plot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    # ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    # ax1.plot(df.index, df.Close)
    # ax2.bar(df.index, df.Volume)
    # plt.show()


def observations_par_class(true_labels):
    y_pos = np.arange(len(classes))
    quantities = [true_labels.count('Up'), true_labels.count('Same'), true_labels.count('Down')]
    plt.figure()
    bar = plt.bar(y_pos, quantities, align='center', alpha=0.5)
    bar[0].set_color('g')
    bar[1].set_color('black')
    bar[2].set_color('r')
    plt.xticks(y_pos, classes)
    plt.ylabel('Quantities of observations')
    plt.xlabel('Classes')
    plt.title('Number of observations per classes')
    plt.show(block=False)

    quantities_prc = [100 * true_labels.count('Up') / len(true_labels), 100 * true_labels.count('Same') /
                      len(true_labels), 100 * true_labels.count('Down') / len(true_labels)]
    plt.figure()
    bar = plt.bar(y_pos, quantities_prc, align='center', alpha=0.5)
    bar[0].set_color('g')
    bar[1].set_color('black')
    bar[2].set_color('r')
    plt.xticks(y_pos, classes)
    plt.ylabel('Percent of observations')
    plt.xlabel('Classes')
    plt.title('Percent of observations per classes')
    plt.show()
