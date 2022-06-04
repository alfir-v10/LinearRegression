import argparse
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


def plot(x, y):
    plt.scatter(x, y)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.title(' price of a car for a given mileage')
    plt.show()


def gradient_descent(y_pred, y, x, lr, w):
    w[0] -= lr / y.shape[0] * np.sum(y_pred - y)
    w[1] -= lr / y.shape[0] * np.sum((y_pred - y) * x)
    return w


def predict(w, x):
    return w[0] + w[1] * x


def mean_square_error(y, y_pred):
    return np.sum((y - y_pred) ** 2) / y.shape[0]


def mean_absolute_error(y, y_pred):
    return np.sum(abs(y - y_pred)) / y.shape[0]


def MinMaxScaler(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def MinMaxScalerReverse(x):
    return (x.max() - x.min()) * x + x.min()


def StandartScaler(x, x_mean=None, x_std=None):
    if x_mean is None:
        x_mean = x.mean()
    if x_std is None:
        x_std = x.std()
    return (x - x_mean) / x_std


def StandartScalerReverse(x):
    return x * x.std() + x.mean()


def save_model(w, x, y, path):
    with open(path, 'w') as f:
        for i, k in enumerate(w):
            f.write(f'w{i}: {k}\n')
        f.write(f'x_max: {x.max()}\n')
        f.write(f'x_min: {x.min()}\n')
        f.write(f'x_mean: {x.mean()}\n')
        f.write(f'x_std: {x.std()}\n')


def load_model(path):
    with open(path, 'r') as f:
        w = [line.rstrip().split(':') for line in f.readlines()]
        w = {k: float(v) for k, v in w}
    return w


def train(path):
    df = pd.read_csv(path)
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    # plot(x_data, y_data)

    # x = StandartScaler(x_data)
    x = MinMaxScaler(x_data)

    # y = StandartScaler(y_data)
    # y = MinMaxScaler(y_data)

    lr = 0.01
    w = np.zeros(2)

    mse = []
    mae = []
    for _ in range(200000):
        y_pred = predict(w, x)
        w = gradient_descent(y_pred, y_data, x, lr, w)
        mse.append(mean_square_error(y_data, y_pred))
        mae.append(mean_absolute_error(y_data, y_pred))
    save_model(w, x_data, y_data, 'weights.txt')

    """MSE"""
    plt.plot(range(len(mse)), mse, color='red')
    plt.title('MSE by iter')
    plt.xlabel('iter')
    plt.ylabel('error')
    plt.savefig('mse.png')
    plt.show()


    """MAE"""
    plt.plot(range(len(mae)), mae, color='green')
    plt.title('MAE by iter')
    plt.xlabel('iter')
    plt.ylabel('error')
    plt.savefig('mae.png')
    plt.show()


    """show result of linear regression"""
    plt.scatter(x_data, y_data, label='real', marker='*', color='green')
    plt.plot(x_data, y_pred, label='predict')
    plt.legend()
    plt.title('Result of linear regression')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.savefig('result.png')
    plt.show()


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('dataset', help='data.csv')
    train(pars.parse_args().dataset)
