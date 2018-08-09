from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from model_evaluation import evaluate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


def predict_df(data, Y):
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    # classifiers = [DecisionTreeClassifier(criterion="entropy"), GaussianNB(), SVC(random_state=2)]
    # classifiers_type = ["Decision Tree", "Gaussian Naive Bayes", "SVC"]
    classifiers = [DecisionTreeClassifier(criterion="entropy")]
    classifiers_type = ["Decision Tree"]
    for clf, classifiers_type in zip(classifiers, classifiers_type):
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        evaluate(clf, classifiers_type, X_test, y_test, X_train, y_train, predictions)

