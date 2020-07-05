import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras import layers
from time import time


def linear_regression(x, y, test_size=0.3, log_result=False):
    X_train, X_test, y_train, y_test = train_test_split(
        x.values, y.values, test_size=test_size, random_state=24)

    linReg = LinearRegression()
    start = time()
    linReg.fit(X_train, y_train)
    end_train = time()

    y_pred = linReg.predict(X_test).round()
    end_pred = time()

    print(linReg.score(X_test, y_test))
    if log_result:
        for i in np.arange(len(y_pred)):
            print('Actual: ', y_test[i], ', Predicted: ', y_pred[i])
    return [r2_score(y_test, y_pred), end_train -
            start, end_pred - end_train]


def logistic_regression(x, y, test_size=0.3, max_iter=1000, log_time=True):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=24)

    logReg = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', random_state=24, max_iter=max_iter)
    start = time()
    logReg.fit(X_train, y_train)
    end_train = time()

    y_pred = logReg.predict(X_test)
    end_pred = time()

    return [accuracy_score(y_test, y_pred), confusion_matrix(
        y_test, y_pred), end_train - start, end_pred - end_train]


def knn(x, y, test_size=0.3, n_neighbors=33, log_time=True):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=24)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    start = time()
    knn.fit(X_train, y_train)
    end_train = time()

    y_pred = knn.predict(X_test)
    end_pred = time()

    return [accuracy_score(y_test, y_pred), confusion_matrix(
        y_test, y_pred), end_train - start, end_pred - end_train]


def decision_tree(x, y, test_size=0.3, log_time=True):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=24)

    dect = DecisionTreeClassifier(
        criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=24)
    start = time()
    dect.fit(X_train, y_train)
    end_train = time()

    y_pred = dect.predict(X_test)
    end_pred = time()

    return [accuracy_score(y_test, y_pred), confusion_matrix(
        y_test, y_pred), end_train - start, end_pred - end_train]


def neural_network(x, y, test_size=0.3, log_result=False, is_regression=False, epochs=50):
    X_train, X_test, y_train, y_test = train_test_split(
        x.values, y.values, test_size=0.3, random_state=24)
    if not is_regression:
        y_train = np.array([[0] if x == 0 else [1] for x in y_train])
        y_test = np.array([[0] if x == 0 else [1] for x in y_test])
        model = Sequential(
            [
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=["accuracy"])
    else:
        model = Sequential(
            [
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ]
        )
        model.compile(optimizer='adam',
                      loss='mse', metrics=['mae'])
    start = time()
    history = model.fit(X_train, y_train, batch_size=64,
                        epochs=epochs, verbose=0)
    end_train = time()

    # perform prediction
    y_pred = model.predict(X_test)
    end_pred = time()

    # perform auto-evaluation
    if not is_regression:
        if log_result:
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print('Loss = ', loss, ', Accuracy = ', accuracy)
            for i in np.arange(len(y_pred)):
                print('Actual: ', y_test[i], ', Predicted: ', y_pred[i])
        return [accuracy_score(y_test, y_pred.round()), confusion_matrix(
            y_test, y_pred.round()), end_train - start, end_pred - end_train]
    else:
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print('Loss = ', loss, ', Mae = ', mae)
        if log_result:
            for i in np.arange(len(y_pred)):
                print('Actual: ', y_test[i], ', Predicted: ', y_pred[i])
        fig = go.Figure()
        fig.add_trace(go.Scattergl(y=history.history['loss'],
                                   name='Loss'))
        fig.add_trace(go.Scattergl(y=history.history['mae'],
                                   name='Mae'))
        fig.update_layout(height=500, width=700,
                          xaxis_title='Epoch',
                          yaxis_title='Loss')
        fig.show()
        return [r2_score(y_test, y_pred), end_train -
                start, end_pred - end_train]
