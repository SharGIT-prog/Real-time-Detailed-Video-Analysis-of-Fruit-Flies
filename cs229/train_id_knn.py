from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from cs229.load_data_id import X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import get_file

import joblib


def train(X, y):
    # Use a fixed split so the experiment is reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    # KNN with feature standardization
    clf = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test


def report_results(clf, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(
        y_train, clf.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test, clf.predict(X_test)
    )

    print('KNN Results')
    print('-----------')
    print('Train error: {:0.1f}%'.format(
        100 * (1 - train_accuracy)
    ))

    print('Test error: {:0.1f}%'.format(
        100 * (1 - test_accuracy)
    ))

    print('N_train: {}'.format(X_train.shape[0]))
    print('N_test: {}'.format(X_test.shape[0]))


def main():
    X = joblib.load(
        get_file('output', 'data', X_JOBLIB_NAME)
    )

    y = joblib.load(
        get_file('output', 'data', Y_JOBLIB_NAME)
    )

    clf, X_train, X_test, y_train, y_test = train(X, y)

    report_results(
        clf,
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == '__main__':
    main()