from feature_generator import *
from data_visualization import *
from predit_model import *


def main():
    df, true_label = csv_to_pd()
    predict_df(df, true_label)
    plot_figure(df)
    observations_par_class(true_label)


if __name__ == "__main__":
    print("Start {}".format(str(datetime.now())))
    main()
    print("End {}".format(str(datetime.now())))


