import optparse
from termcolor import colored

from data_visualization import *
from nlp_feature_generation import *
from predit_model import *


def main(opt):

    # Prepare features for nlp mode or historical data
    df, true_label = prepare_nlp_features_reuter(opt) if opt.nlp else csv_to_pd(opt)

    # Predict
    predict_df(df, true_label, opt) if not opt.skip_predict else None

    # Visualize data
    visualization(df, true_label, opt) if opt.data_visualization else None
    exit(0)

'''
    :description - this function create log file that will used to print the 
    results of the experiments.
    :param options - holds options to run including: parameter_tuning, windows_mode,
     enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
'''


def create_log_file(options):
    options_str = "nlp-{}_ptuning-{}_window-{}_eval-{}_" \
                  "visual-{}_skipP-{}_vix-{}_dbg-{}"\
        .format(options.nlp, options.parameter_tuning, options.windows_mode,
                options.evaluation_mode, options.data_visualization,
                options.skip_predict, options.enable_vix, options.debug)
    logger_file_name = 'logger_{}.log'.format(options_str)
    log_file = os.path.join(os.path.dirname(__file__), "LOGS", logger_file_name)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    with open(log_file, 'w'):
        pass
    print("Start {}, log file: {}".format(str(datetime.now()), log_file))
    logger.info("Start {}".format(str(datetime.now())))


if __name__ == "__main__":
    # Define options
    parser = optparse.OptionParser()
    parser.add_option('--parameter_tuning', action="store_true", dest="parameter_tuning",
                      help="<Optional> Parameter tuning", default=False)
    parser.add_option('--windows_mode', choices=["one", "multiple", "sliding"],
                      dest="windows_mode",
                      help="<Optional> In hyper-parameter tuning choose a window type, Default - one"
                      " one - for one window, all the data in the same window"
                      " multiple - for windows without intersection"
                      " sliding - for windows with intersection"
                      , default="one")
    parser.add_option('--enable_vix', action="store_true", dest="enable_vix",
                      help="<Optional> enable vix features", default=False)
    parser.add_option('--nlp', dest="nlp", choices=["get_news", "generate_features", "full_flow", "minimal"],
                      help="<Optional> NLP features: get_news for web-scraping, "
                      "generate_features for generating the features"
                      "full_flow for running with web-scaping and generate news"
                      "minimal is used for loading features from file, and NOT scraping news"
                      , default=False)
    parser.add_option('--debug', action="store_true", dest="debug",
                      help="<Optional> For faster results, debug mode", default=False)
    parser.add_option('--data_visualization', action="store_true", dest="data_visualization",
                      help="<Optional> Data analysis visualization", default=False)
    parser.add_option('--skip_predict', action="store_true", dest="skip_predict",
                      help="<Optional> Skip prediction", default=False)
    parser.add_option('--evaluation_mode', choices=["classifier", "major", "weighted"],
                      dest="evaluation_mode", help="<Optional>Evaluation mode", default="classifier")
    options, args = parser.parse_args()
    # Log
    create_log_file(options)
    print("Running with the following parameters {}".format(options))
    logger.info("Running with the following parameters {}".format(options))
    # Main
    try:
        main(options)
    except Exception:
        print(colored('FAILED!', 'red'))
        logger.critical("FAILED!")
    finally:
        print("End {}".format(str(datetime.now())))
        logger.info("End {}".format(str(datetime.now())))


