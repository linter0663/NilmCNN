from tensorflow.keras.backend import mean, equal, round, abs, sum, epsilon


def soft_acc(y_true, y_pred):
    return mean(equal(round(y_true), round(y_pred)))


def estimated_accuracy(y_true, y_pred):
    acc = 1 - sum(abs(y_pred - y_true)) / (2 * (sum(y_true) + epsilon()))
    if acc < 0:
        acc = 0.0
    return acc
