from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, make_scorer, recall_score
import matplotlib.pyplot as plt
import itertools
from config import *

#  In this file there is everything related to evaluation


'''
    :description - calculate the evaluated score according evaluation_mode
    :param true - the true labels of data 
    :param predictions - the predicated labels of data
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode  
    :returns - the evaluated score according evaluation_mode
'''


def evaluate_by_choice(true, predictions, opt):
    # normal accuracy_score of classifier
    if opt.evaluation_mode == "classifier":
        return "classifier", accuracy_score(y_true=true, y_pred=predictions)
    # Evaluation of worst "false negative" (e.g. saying up instead of down)
    elif opt.evaluation_mode == "major":
        return "major", major_fn_acc_calc(true, predictions)
    # weighted error
    elif opt.evaluation_mode == "weighted":
        return "weighted", weighted_acc(true, predictions)

'''
    :description - calculate train and test accuracy by evaluation mode,
    plot confusion matrix, calculate base accuracy for one class classifier.
    Also calculate f1, recall and precision (not in used).
    :param classifiers_type - Classifier type (indicating if there is a limited
    complexity)
    :param y_test - the true labels of test 
    :param y_train - the true labels of train 
    :param predictions - the predicated labels of test  
    :param predictions_train - the predicated labels of train
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode   
'''


def evaluate(classifiers_type, y_test, y_train, predictions, predictions_train, opt):
    print("Start evaluate, time {}".format(str(datetime.now())))
    logger.info("Start evaluate, time {}".format(str(datetime.now())))
    print("Evaluating classifier {}".format(classifiers_type))
    logger.info("Evaluating classifier {}".format(classifiers_type))

    # Print accuracy train
    count_predictions(predictions) if opt.debug else None
    eval_type, train_accuracy = evaluate_by_choice(y_train, predictions_train, opt)
    print("Model TRAIN \'{}\' evaluation: {:.2f}%".format(eval_type, 100 * train_accuracy))
    logger.info("Model TRAIN \'{}\' evaluation: {:.2f}%".format(eval_type, 100 * train_accuracy))

    # Print accuracy test
    eval_type, test_accuracy = evaluate_by_choice(y_test, predictions, opt)
    print("Model TEST \'{}\' evaluation: {:.2f}%".format(eval_type, 100 * test_accuracy))
    logger.info("Model TEST \'{}\' evaluation: {:.2f}%".format(eval_type, 100 * test_accuracy))

    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, predictions, classes)
    precision = precision_score(y_test, predictions, labels=classes, average=None)
    recall = recall_score(y_test, predictions, labels=classes, average=None)
    f1score = f1_score(y_test, predictions, labels=classes, average=None)
    for value, elem in [(precision, "precision"), (recall, "recall"), (f1score, "f1score")]:
        for class_name, val in zip(classes, value):
            print("TEST {}-{}: {:.2f}%".format(class_name, elem, 100 * val))
            logger.info("TEST {}-{}: {:.2f}%".format(class_name, elem, 100 * val))
    plot_confusion_matrix(cnf_matrix, classes, classifiers_type, opt)

    # Calculate base accuracy
    for class_name in classes:
        eval_type, base_case_acc = calculate_base_accuracy(y_test, class_name, opt)
        print("Base Model TEST {}-\'{}\' evaluation: {:.2f}%".format(class_name, eval_type, 100 * base_case_acc))
        logger.info("Base Model TEST {}-\'{}\' evaluation: {:.2f}%".format(class_name, eval_type, 100 * base_case_acc))
    print("===========================================================================\n")
    logger.info("===========================================================================\n")

'''
    :description - this function plot nicely the confusion matrix.
    :param cm - confusion matrix
    :param classes - Same, Up, Down classifications
    :param classifier_name - the classifier name (including limited complexity
     if exists)
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode
    :param normalize - default value false, option to normalized the value in the
    confusion matrix
    :param title - default value 'Confusion matrix', holds the title of the image 
    :param cmap  - default value plt.cm.Blues, related to colors in the image
'''


def plot_confusion_matrix(cm, classes, classifier_name, opt, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, classifier {}".format(classifier_name))
        logger.info("Normalized confusion matrix, classifier {}".format(classifier_name))
    else:
        print("Confusion matrix, without normalization, classifier {}".format(classifier_name))
        logger.info("Confusion matrix, without normalization, classifier {}".format(classifier_name))
    print(cm)
    logger.info("\n{}".format(cm))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = title + "_classifier_" + classifier_name + "_evaluation_" + \
            opt.evaluation_mode + "_windows_" + opt.windows_mode
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show(block=False)
    path = os.path.join(os.getcwd(), "Confusion_matrix") if not opt.nlp \
        else os.path.join(os.getcwd(), "NLP", "Confusion_matrix")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if not normalize:
        plt.savefig(os.path.join(path, "{}.png".format(title)))
    else:
        plt.savefig(os.path.join(path, "Normalized_{}.png".format(title)))

'''
    :description - calculate accuracy according major evaluation mode.
    Meaning only errors that the prediction label opposite to the true label
    are counted. (Up label vs Down label and otherwise).
    :param y - true labels of data 
    :param predictions - predicated labels of data
    :returns the calculated accuracy
'''


def major_fn_acc_calc(y, predictions):
    count = 0
    for test, pred in zip(y, predictions):
        count += 1 if (test == "Up" and pred == "Down") or (test == "Down" and pred == "Up") else 0
    err = count / len(y)
    return 1 - err  # Accuracy is 1 - error

'''
    :description - calculate accuracy according minor evaluation.
    Meaning only errors that the prediction label not that far from the true label
    are counted. For example: same label vs up label, same label vs down label
    and otherwise.
    :param y - true labels of data 
    :param predictions - predicated labels of data
    :returns the calculated accuracy
'''


def minor_fn_acc_calc(y, predictions):
    count = 0
    for test, pred in zip(y, predictions):
        count += 1 if (test == "Up" and pred == "Same") or (test == "Down" and pred == "Same") or \
            (test == "Same" and pred == "Up") or (test == "Same" and pred == "Down") else 0
    err = count / len(y)
    return 1 - err  # Accuracy is 1 - error

'''
    :description - calculate the accuracy by weighted mode. Alpha weight multiply
    the major accuracy plus 1 minus alpha multiply the minor accuracy. 
    :param y - the true labels
    :param predictions - the predicated labels
    :returns - the accuracy by weighted mode.
'''


def weighted_acc(y, predictions):
    return alpha * major_fn_acc_calc(y, predictions) + (1 - alpha) * minor_fn_acc_calc(y, predictions)

'''
    :description - the functions calculate and printed the number for each label
    in the given input and the total of predictions also.
    :param predictions - the predicated labels.
'''


def count_predictions(predictions):
    count = [0, 0, 0]  # count up, same, down
    for pred in predictions:
        count[0] += 1 if pred == "Up" else 0
        count[1] += 1 if pred == "Same" else 0
        count[2] += 1 if pred == "Down" else 0
    for idx, value in enumerate(count):
        label = "Up" if idx == 0 else "Same" if idx == 1 else "Down"
        print("Total {}: {}".format(label, value))
        logger.info("Total {}: {}".format(label, value))

'''
    :description - this function calculate the accuracy if we will classify all 
    data to the same class.
    :param y_test - the true labels
    :param direction - the class which will classify to.
    :param opt - holds options to run including: parameter_tuning, windows_mode,
    enable_vix, nlp, debug, data_visualization, skip_predict, evaluation_mode 
    :returns - the accuracy if we will classify everything to the same class.
'''


def calculate_base_accuracy(y_test, direction, opt):
    predictions = [direction for _ in enumerate(y_test)]
    return evaluate_by_choice(y_test, predictions, opt)



