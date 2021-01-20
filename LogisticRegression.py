import tensorflow as tf
import warnings
import numpy as np
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")


class LogisticRegression:
    def __init__(self, lr=0.1, num_iter=100, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def plotfunction(train_accs, dev_accs):
    plt.rcParams['figure.figsize'] = [15, 10]

    ax = plt.subplot(111)
    t1 = np.arange(0.0, 1.0, 0.01)
    plt.plot(list(range(len(train_accs))), train_accs, '^', label="TRAIN")
    plt.plot(list(range(len(dev_accs))), dev_accs, label="DEV")
    leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()

    ax = plt.subplot(111)
    t1 = np.arange(0.0, 1.0, 0.01)
    plt.plot(list(range(len(train_accs))), train_accs, '^', label="TRAIN")
    plt.plot(list(range(len(dev_accs))), dev_accs, label="DEV")
    leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.ylim(0.0, 1.1)
    plt.show()


def run():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(  # x  = kritiki y = pos/neg
        path="imdb.npz",
        num_words=10000,
        skip_top=100,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3,
    )
    data = np.concatenate((x_train, x_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)
    data = vectorize(data)
    targets = np.array(targets).astype("int64")

    test_x = data[:25000]
    test_y = targets[:25000]
    train_x = data[25000:]
    train_y = targets[25000:]
    train_accs = []
    dev_accs = []

    for i in range(1, 11):
        test_x = data[:25000]
        test_y = targets[:25000]
        p = round((50000 - 25000) * (i / 10))
        end_train = 25000 + p
        train_x = data[25000:end_train]
        train_y = targets[25000:end_train]
        model = LogisticRegression(fit_intercept=False)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        y_pred_train = model.predict(train_x)
        acc = accuracy(test_y, y_pred)
        acc_train = accuracy(train_y, y_pred_train)
        #print("Accuracy:", acc)
        #print("Accuracy in train set", acc_train)
        train_accs.append(acc_train)
        dev_accs.append(acc)

    plotfunction(train_accs, dev_accs)


if __name__ == '__main__':
    start = time.time()
    print("Timer Started Now")
    run()
    end = time.time()
    print("Timer Ended Now")
    timeEl = (end - start) / 60
    print("Time Elapsed : ", timeEl, "Minutes")
