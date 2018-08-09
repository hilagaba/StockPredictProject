from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np
from config import *


def evaluate(classifier, classifiers_type, X_test, y_test, X_train, y_train, predictions):
    # train_accuracy = classifier.score(X_train, y_train)
    train_accuracy = accuracy_score(y_true=y_train, y_pred=classifier.predict(X_train))
    print("Classifier {}".format(classifiers_type))
    print("Model TRAIN accuracy: {0:.2f}%".format(100 * train_accuracy))
    # model_accuracy = classifier.score(X_test, y_test)
    model_accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    from data_visualization import observations_par_class
    observations_par_class(y_test)
    exit(555345)
    print("Model total accuracy: {0:.2f}%".format(100 * model_accuracy))
    cnf_matrix = confusion_matrix(y_test, predictions, classes)
    plot_confusion_matrix(cnf_matrix, classes, classifiers_type)
    # true = [2 if x == 'Up' else 0 if x == "Down" else 1 for x in Y]
    # pred = [2 if x == 'Up' else 0 if x == "Down" else 1 for x in all_pred]
    # for i in range(1, len(classes)):
    #     j = i - 1
    #     roc(true, pred, j, i)
    #     roc(true, pred, i, j)
    print("===========================================================================\n")


def plot_confusion_matrix(cm, classes, classifier_name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, classifier {}".format(classifier_name))
    else:
        print("Confusion matrix, without normalization, classifier {}".format(classifier_name))
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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
    if not os.path.exists(os.path.join(os.getcwd(), "Confusion_matrix")):
        os.mkdir(os.path.join(os.getcwd(), "Confusion_matrix"))
    if not normalize:
        plt.savefig(os.path.join(os.getcwd(), "Confusion_matrix") + "\\confusion_matrix_{}.png".format(classifier_name))
    else:
        plt.savefig(os.path.join(os.getcwd(), "Confusion_matrix") +
                    "\\confusion_matrix_normalize_{}.png".format(classifier_name))


# def roc(true_labels, prediction, true_val, false_val):
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(true_labels, prediction, pos_label=true_val)
#     roc_auc = auc(fpr, tpr)
#     # Plot ROC curve
#     plt.clf()
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic {} to {}'.format(str(true_val), str(false_val)))
#     plt.legend(loc="lower right")
#     plt.show(block=False)
#     if not os.path.exists(os.path.join(os.getcwd(), "ROC")):
#         os.mkdir(os.path.join(os.getcwd(), "ROC"))
#     plt.savefig(os.path.join(os.getcwd(), "ROC") + "\\roc_{}_to_{}.png".format(str(true_val), str(false_val)))
