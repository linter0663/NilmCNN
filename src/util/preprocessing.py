import numpy as np


def split_data(aggregate, appliances, validation_split, evaluate_split):
    bins = aggregate.shape[0]

    evalute_splices = int(bins * evaluate_split)
    validation_splices = int(bins * validation_split)

    x_eval = aggregate[(bins - evalute_splices):bins]
    x_train = aggregate[(bins - (evalute_splices + validation_splices)):(bins - evalute_splices)]

    y_eval = appliances[(bins - evalute_splices):bins]
    y_train = appliances[(bins - (evalute_splices + validation_splices)):(bins - evalute_splices)]

    return x_train, y_train, x_eval, y_eval


def process_data(data):
    apps_len = data.shape[1]
    bins = data.shape[0] // 1440

    new_size = bins * 1440

    aggregate = np.empty((new_size, 1, 1))
    appliances = np.empty((new_size, data.shape[1] - 2, 1))

    for i in range(new_size):
        if data[i][1][0] == 0:
            aggregate[i] = 1
            appliances[i] = np.zeros((data.shape[1] - 2, 1))
        else:
            aggregate[i] = 1 - (sum(data[i:i + 1:, 2:apps_len, 0:0 + 1][0]) / data[i][1][0])
            appliances[i] = data[i:i + 1, 2:apps_len, 0:0 + 1][0] / data[i][1][0]

    aggregate = (np.asarray(aggregate)).reshape((bins, 1440, 1, 1))
    appliances = (np.asarray(appliances)).reshape((bins, 1440, data.shape[1] - 2, 1))

    return aggregate, appliances
