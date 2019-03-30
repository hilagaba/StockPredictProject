from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from model_evaluation import evaluate, evaluate_by_choice  # , roc_generator
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from config import *
from statistics import mean

# This file responsible for prediction

'''
    :description - this is the main function in the prediction model.
    Here we execute different type of predictions (with and without hyper 
    parameter tuning). And of course run the best prediction according
    to the hyper parameter tuning phase.
    :param data - data frame that hold all data's features 
    :param Y - the true label corresponding to each row in data
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
'''


def predict_df(data, Y, opt):
    X = data.values
    classifiers = [[DecisionTreeClassifier(criterion="entropy"), "Decision Tree"]]
    without_params(X, Y, classifiers, opt, features=list(data.keys()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    if opt.parameter_tuning:
        # Find best parameter
        optimizer_type, optimizer_value = parameters_tuning(opt, X_train, y_train, classifiers, data.Close)
        # Run prediction with the best limitation
        run_best_optimizer(classifiers, optimizer_type, optimizer_value, X_train, X_test, y_train, y_test, opt, features=list(data.keys()))

'''
    :description - this function preform hyper parameter tuning. By running
    predictions by train-validation structure, including iterating a wide
    range of value for 3 limited parameters of the decision tree.
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param X_train - the features for the train data set
    :param classifiers - the classifiers list (in our case it's only decision tree)
    :param data_close - the column of closing price for each day. Used for getting
    the length of data.
    :returns optimizer_type - the best limited parameter name 
    :returns optimizer_value - the best limited parameter value
'''


def parameters_tuning(opt, X_train, y_train, classifiers, data_close):
    print("================= Hyper-parameters tuning =================")
    logger.info("================= Hyper-parameters tuning =================")

    optimizer_type, optimizer_value, best_acc_val, best_acc_train = "", 0, 0, 0
    # Min Samples
    print("Start min samples for split, time {}".format(str(datetime.now())))
    logger.info("Start min samples for split, time {}".format(str(datetime.now())))
    iterate_range = range(2, 10) if opt.debug else range(2, len(data_close), complexity_steps) if not opt.windows_mode == "sliding" and not opt.windows_mode == "multiple" else range(2, window_size, complexity_steps)
    opt_min_samples, max_validation, max_train = min_samples_for_split(X_train, y_train, classifiers, min_samples_splits=iterate_range, opt=opt)
    if max_validation > best_acc_val or max_validation == best_acc_val and max_train > best_acc_train:
        optimizer_type = "min_samples_split"
        optimizer_value = opt_min_samples
        best_acc_val = max_validation
        best_acc_train = max_train
    classifiers[0][0].set_params(max_depth=None, min_samples_split=2, max_leaf_nodes=None)

    # Max Depth
    print("Start max depth, time {}".format(str(datetime.now())))
    logger.info("Start max depth, time {}".format(str(datetime.now())))
    max_depths = range(1, 10) if opt.debug else range(1, 100, 1)
    opt_max_depth, max_validation, max_train = max_depth(X_train, y_train, classifiers, max_depths=max_depths, opt=opt)
    if max_validation > best_acc_val or max_validation == best_acc_val and max_train > best_acc_train:
        optimizer_type = "max_depth"
        optimizer_value = opt_max_depth
        best_acc_val = max_validation
        best_acc_train = max_train
    classifiers[0][0].set_params(max_depth=None, min_samples_split=2, max_leaf_nodes=None)

    # Max number of leaf
    print("Start max number of leaf, time {}".format(str(datetime.now())))
    logger.info("Start max number of leaf, time {}".format(str(datetime.now())))
    leafs_range = range(2, 10) if opt.debug else range(2, 1000, complexity_steps) if not opt.windows_mode == "sliding" else range(2, window_size, complexity_steps)
    opt_max_leaf, max_validation, max_train = max_num_of_leafs(X_train, y_train, classifiers, max_leafs=leafs_range, opt=opt)
    if max_validation > best_acc_val or max_validation == best_acc_val and max_train > best_acc_train:
        optimizer_type = "max_leaf_nodes"
        optimizer_value = opt_max_leaf
        best_acc_val = max_validation
        best_acc_train = max_train
    classifiers[0][0].set_params(max_depth=None, min_samples_split=2, max_leaf_nodes=None)
    print("Train accuracy {:.2f}% Validation accuracy {:.2f}% \nOptimizer {} Optimizer parameter {}"
          .format(best_acc_train, best_acc_val, optimizer_type, optimizer_value))
    logger.info("Train accuracy {:.2f}% Validation accuracy {:.2f}% \nOptimizer {} Optimizer parameter {}"
                .format(best_acc_train, best_acc_val, optimizer_type, optimizer_value))
    return optimizer_type, optimizer_value

'''
    :description - this function perform train on the train data and predict
    on test data and later evaluate the results. There is option here to different
    windows mode. Also there is a tree visualization option.
    :param X - features
    :param Y - true labels
    :param classifiers - the classifiers list (in our case it's only decision tree)
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param features - features' names
'''


def without_params(X, Y, classifiers, opt, features):
    print("================= Without params =================")
    logger.info("================= Without params =================")

    for (clf, classifiers_type) in classifiers:
        predictions = []
        step, window_size_mode = get_step_and_window_size(opt, len(X))
        pred_train, pred_test = [], []
        windows_num = 0
        for i in range(0, len(X), step):
            windows_num += 1
            X_window = X[i: i + window_size_mode]
            Y_window = Y[i: i + window_size_mode]
            X_train, X_test, y_train, y_test = train_test_split(X_window, Y_window, test_size=0.2, shuffle=False)
            clf.fit(X_train, y_train)
            predictions_train = clf.predict(X_train)
            predictions_test = clf.predict(X_test)
            predictions.append(predictions_train)
            predictions.append(predictions_test)
            _, train_accuracy = evaluate_by_choice(y_train, predictions_train, opt)
            _, test_accuracy = evaluate_by_choice(y_test, predictions_test, opt)
            pred_train.append(train_accuracy)
            pred_test.append(test_accuracy)
            if i + window_size_mode - 1 >= len(X):
                break
        if opt.windows_mode in ("multiple", "sliding"):
            print("TRAIN Average {} evaluation {:.2f}%".format(opt.evaluation_mode, 100 * mean(pred_train)))
            print("TEST Average {} evaluation {:.2f}%".format(opt.evaluation_mode, 100 * mean(pred_test)))
            logger.info("TRAIN Average {} evaluation {:.2f}%".format(opt.evaluation_mode, 100 * mean(pred_train)))
            logger.info("TEST Average {} evaluation {:.2f}%".format(opt.evaluation_mode, 100 * mean(pred_test)))
            plot_accuracy(range(0, windows_num), pred_train, pred_test,
                          x_label='Window number', title='Train and Test accuracy {} evaluation'.format(opt.evaluation_mode),
                          opt=opt)
        else:
            tree_visualization(clf, classifiers_type, opt, feature_list=features)
            evaluate(classifiers_type, y_test, y_train, predictions_test, predictions_train, opt)

'''
    :description - this function find the best accuracy on depth, using hyper-parameter
    tuning depth.
    :param X - features
    :param Y - true label
    :param classifiers - classifier type
    :param max_depths - depth range, hyper parameter tuning 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns complexity value, 
    :returns max validation accuracy
    :returns max_train accuracy
'''


def max_depth(X, Y, classifiers, max_depths, opt):
    train_acc_list, validation_acc_list = [], []
    for clf in classifiers:
        for max_depth in max_depths:
            clf[0].set_params(max_depth=max_depth)
            clf[1] = "Decision Tree_depth-{}".format(max_depth)
            train_acc, validation_acc = get_accuracy_by_classifier(clf[0], X, Y, opt)
            train_acc_list.append(train_acc)
            validation_acc_list.append(validation_acc)
        plot_accuracy(max_depths, train_acc_list, validation_acc_list, x_label='Depth',
                      title="Accuracy depending on tree depth by {} evaluation".format(opt.evaluation_mode), opt=opt)
        return find_opt_complexity(max_depths, train_acc_list, validation_acc_list)


'''
    :description - this function find the best accuracy depend on leafs, using hyper-parameter
    tuning number of leafs.
    :param X - features
    :param Y - true label
    :param classifiers - classifier type
    :param max_num_of_leafs - leaf range, hyper parameter tuning 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns complexity value, 
    :returns max validation accuracy
    :returns max_train accuracy
'''


def max_num_of_leafs(X, Y, classifiers, max_leafs, opt):
    train_acc_list, validation_acc_list = [], []
    for clf in classifiers:
        for max_leaf in max_leafs:
            print("max number LEAF {}, time {}".format(max_leaf, str(datetime.now()))) if max_leaf % 100 == 0 else None
            logger.info("max number LEAF {}, time {}".format(max_leaf, str(datetime.now()))) if max_leaf % 100 == 0 else None
            clf[0].set_params(max_leaf_nodes=max_leaf)
            clf[1] = "Decision Tree_leaf-{}".format(max_leaf)
            train_acc, validation_acc = get_accuracy_by_classifier(clf[0], X, Y, opt)
            train_acc_list.append(train_acc)
            validation_acc_list.append(validation_acc)
        plot_accuracy(max_leafs, train_acc_list, validation_acc_list, x_label='Leaf',
                      title="Accuracy depending on tree leaf by {} evaluation".format(opt.evaluation_mode), opt=opt)
        return find_opt_complexity(max_leafs, train_acc_list, validation_acc_list)


'''
    :description - this function find the best accuracy depend on samples for split, 
    using hyper-parameter tuning minimum number samples for split.
    :param X - features
    :param Y - true label
    :param classifiers - classifier type
    :param min_samples_splits - minimum samples to split range, hyper parameter tuning 
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns complexity value, 
    :returns max validation accuracy
    :returns max_train accuracy
'''


def min_samples_for_split(X, Y, classifiers, min_samples_splits, opt):
    train_acc_list, validation_acc_list = [], []
    for clf in classifiers:
        for min_samples_split in min_samples_splits:
            print("LEAF {}, time {}".format(min_samples_split, str(datetime.now()))) if min_samples_split % 1000 == 0 else None
            logger.info("LEAF {}, time {}".format(min_samples_split, str(datetime.now()))) if min_samples_split % 1000 == 0 else None

            clf[0].set_params(min_samples_split=min_samples_split)
            clf[1] = "Decision Tree_min_samples_split-{}".format(min_samples_split)
            train_acc, validation_acc = get_accuracy_by_classifier(clf[0], X, Y, opt)
            train_acc_list.append(train_acc)
            validation_acc_list.append(validation_acc)
        plot_accuracy(min_samples_splits, train_acc_list, validation_acc_list,
                      x_label='Minimum samples for split',
                      title="Accuracy depending on samples split by {} evaluation".format(opt.evaluation_mode), opt=opt)
        return find_opt_complexity(min_samples_splits, train_acc_list, validation_acc_list)


'''
    :description - this function find the accuracy of the classifier on train and validation
    :param X - features
    :param Y - true label
    :param clf - classifier  
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns avg_pred_train - average prediction on train accuracy
    :returns avg_pred_test - average prediction on test accuracy
'''


def get_accuracy_by_classifier(clf, X, Y, opt):
    step, window_size_mode = get_step_and_window_size(opt, len(X))
    sum_pred_train, sum_pred_test, count_pred = 0, 0, 0
    for i in range(0, len(X), step):
        X_window = X[i: i + window_size_mode]
        Y_window = Y[i: i + window_size_mode]
        X_train, X_test, y_train, y_test = train_test_split(X_window, Y_window, test_size=0.2, shuffle=False)
        clf.fit(X_train, y_train)
        predictions_train = clf.predict(X_train)
        predictions_test = clf.predict(X_test)
        _, train_accuracy = evaluate_by_choice(y_train, predictions_train, opt)
        _, test_accuracy = evaluate_by_choice(y_test, predictions_test, opt)
        sum_pred_train += train_accuracy
        sum_pred_test += test_accuracy
        count_pred += 1
        if i + window_size_mode - 1 >= len(X):
            break
    avg_pred_train = sum_pred_train/count_pred
    avg_pred_test = sum_pred_test / count_pred
    return avg_pred_train, avg_pred_test

'''
    :description - this function plots the accuracy of train and validation depend on parameter
    :param x_values - features
    :param x_label - the parameter
    :param train_acc_list - list of train accuracy
    :param validation_acc_list - list of validation accuracy
    :param title - chart title
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns None, save charts to the suitable directory
'''


def plot_accuracy(x_values, train_acc_list, validation_acc_list, x_label,
                  title, opt):
    is_window = opt.windows_mode == "multiple"
    is_sliding = opt.windows_mode == "sliding"
    plt.figure()
    plt.plot(x_values, train_acc_list, color='red', label="Train")
    plt.plot(x_values, validation_acc_list, color='blue', label="Validation")
    plt.xticks()
    plt.ylabel('Accuracy')
    plt.xlabel(x_label)
    title += " - Window" if is_window else " - Sliding" if is_sliding else ''
    plt.title(title)
    plt.legend(loc='best')
    plt.show(block=False)
    tree_parameters_dir = "tree_parameters_windows" if is_window \
        else "tree_parameters_sliding" if is_sliding \
        else "tree_parameters"
    tree_parameters_dir = tree_parameters_dir if not opt.nlp else os.path.join("NLP", tree_parameters_dir)
    if not os.path.exists(os.path.join(os.getcwd(), tree_parameters_dir)):
        os.makedirs(os.path.join(os.getcwd(), tree_parameters_dir), exist_ok=True)
    plt.savefig(os.path.join(os.getcwd(), tree_parameters_dir, "{}.png".format(title)))


'''
    :description - this function plots the tree
    :param clf - classifier
    :param classifiers_type - classifier type
    :param feature_list - features
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :returns None, save charts to the suitable directory
'''


def tree_visualization(clf, classifiers_type, opt, feature_list):
    print("Start tree visualization, time {}".format(str(datetime.now())))
    logger.info("Start tree visualization, time {}".format(str(datetime.now())))
    path = os.path.join(os.getcwd(), "tree_visualization") \
        if not opt.nlp else os.path.join(os.getcwd(), "NLP", "tree_visualization")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_name = "{}_evaluation_{}".format(classifiers_type, opt.evaluation_mode)
    out_file = os.path.join(path, file_name)
    export_graphviz(clf, out_file=out_file, class_names=["Down", "Same", "Up"], filled=True, feature_names=feature_list)
    # Can use online website for visualization from out_file on site: http://webgraphviz.com/
    with open(out_file) as f:
        dot_graph = f.read()
    g = graphviz.Source(dot_graph)
    g.render(filename=file_name, directory=path)


'''
    :description - this function returns step size and windows size
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param len_x: len of X, where X is the dataset
    :returns: step and window size
'''


def get_step_and_window_size(opt, len_x: int)->(int, int):
    if opt.windows_mode == "multiple":
        return window_size, window_size
    elif opt.windows_mode == "sliding":
        return complexity_steps, window_size
    else:
        return len_x, len_x


'''
    :description - this function returns the optimal complexity, maximum train and validation accuracy
    :param complexity: complexity hyper parameter list. list of values of complexity 
    (max_depth, min_samples_split, number of leaf) 
    :param train_acc_list: list of train accuracy
    :param validation_acc_list: list of validation accuracy
    :returns complexity value, 
    :returns max validation accuracy
    :returns max_train accuracy
'''


def find_opt_complexity(complexity, train_acc_list, validation_acc_list):
    max_validation = max(validation_acc_list)
    idx_validation = []
    for i, val in enumerate(validation_acc_list):
        if val == max_validation:
            idx_validation.append(i)
    idx_train = 0
    max_train = 0
    for i in idx_validation:
        if train_acc_list[i] > max_train:
            max_train = train_acc_list[i]
            idx_train = i
    return complexity[idx_train], max_validation, max_train


'''
    :description - running the best optimizer with its best value, setting the value and the optimizer and its value
    on classifier, fitting on train group and evaluating on test group
    :param clf - classifier
    :param optimizer_type - type of optimize: min_samples_split \ max_depth \ max_leaf_nodes 
    :param optimizer_value - the optimal value of the optimizer
    :param X_train - train part of data  
    :param X_test - test part of data
    :param y_train - train part of true labels
    :param y_test - test part of true labels
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param features - list of features label 
    :returns None
'''


def run_best_optimizer(clf, optimizer_type, optimizer_value, X_train, X_test, y_train, y_test, opt, features):
    if optimizer_type == "min_samples_split":
        clf[0][0].set_params(min_samples_split=optimizer_value)
    elif optimizer_type == "max_depth":
        clf[0][0].set_params(max_depth=optimizer_value)
    elif optimizer_type == "max_leaf_nodes":
        clf[0][0].set_params(max_leaf_nodes=optimizer_value)
    clf[0][1] = "Decision Tree_{}-{}".format(optimizer_type, optimizer_value)
    clf[0][0].fit(X_train, y_train)
    predictions_train = clf[0][0].predict(X_train)
    predictions_test = clf[0][0].predict(X_test)
    evaluate(clf[0][1], y_test, y_train, predictions_test, predictions_train, opt)
    tree_visualization(clf[0][0], clf[0][1], opt, feature_list=features)
