from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from cs229.load_data_id import X_JOBLIB_NAME, Y_JOBLIB_NAME
from cs229.files import get_file

import joblib


def report(name, clf, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    print(name)
    print('-' * len(name))
    print('Train error: {:0.1f}%'.format(
        100 * (1 - train_accuracy)
    ))
    print('Test error: {:0.1f}%'.format(
        100 * (1 - test_accuracy)
    ))
    print()


def main():
    X = joblib.load(
        get_file('output', 'data', X_JOBLIB_NAME)
    )

    y = joblib.load(
        get_file('output', 'data', Y_JOBLIB_NAME)
    )

    # ONE split shared by both models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    # Reference model
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs')
    )

    logistic.fit(X_train, y_train)

    # Additional model
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    knn.fit(X_train, y_train)

    print('ID CLASSIFICATION MODEL COMPARISON')
    print('==================================')
    print('N_train:', X_train.shape[0])
    print('N_test:', X_test.shape[0])
    print()

    report(
        'Reference: Logistic Regression',
        logistic,
        X_train, X_test,
        y_train, y_test
    )

    report(
        'Extension: KNN (k=5)',
        knn,
        X_train, X_test,
        y_train, y_test
    )


if __name__ == '__main__':
    main()