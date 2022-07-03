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



def train(path, lr=0.01, epochs=5000,
          path_weight='weights.txt', show_results=False,
          early_stopping=100.0):
    print(path, epochs, early_stopping, path_weight, lr, show_results)
    df = pd.read_csv(path)
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    # plot(x_data, y_data)

    # x = StandartScaler(x_data)
    x = MinMaxScaler(x_data)

    # y = StandartScaler(y_data)
    # y = MinMaxScaler(y_data)

    w = np.zeros(2)


    mse = []
    mae = []
    for _ in range(epochs):
        y_pred = predict(w, x)
        w = gradient_descent(y_pred, y_data, x, lr, w)
        if len(mse) > 10:
            print(mse[-1], mean_square_error(y_data, y_pred), early_stopping)
        if len(mse) > 10 and (mse[-1] - mean_square_error(y_data, y_pred)) < early_stopping:
            print(mse[-1], mean_square_error(y_data, y_pred), early_stopping)
            break
        mse.append(mean_square_error(y_data, y_pred))
        mae.append(mean_absolute_error(y_data, y_pred))

    save_model(w, x_data, y_data, path_weight)

    if show_results:
        """MSE"""
        plt.plot(range(len(mse)), mse, color='red')
        plt.title('MSE by epoch')
        plt.xlabel('epoch')
        plt.ylabel('error')
        # plt.savefig('mse.png')
        plt.show()


        """MAE"""
        plt.plot(range(len(mae)), mae, color='green')
        plt.title('MAE by epoch')
        plt.xlabel('epoch')
        plt.ylabel('error')
        # plt.savefig('mae.png')
        plt.show()


        """show result of linear regression"""
        plt.scatter(x_data, y_data, label='real', marker='*', color='green')
        plt.plot(x_data, y_pred, label='predict')
        plt.legend()
        plt.title(f'Result of linear regression\n lr={lr}; epochs={epochs}')
        plt.xlabel('km')
        plt.ylabel('price')
        plt.savefig('result.png')
        plt.show()


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('--i', help='input file', type=str, default='data.csv')
    pars.add_argument('--lr', help='learning rate', type=float, default=0.01)
    pars.add_argument('--e', help='epochs', type=int, default=5000)
    pars.add_argument('--o', help='output file', type=str, default='weights.txt')
    pars.add_argument('--p', help='plot results', type=bool, default=False)
    pars.add_argument('--es', help='early_stopping', type=float, default=100.0)
    train(path=pars.parse_args().i,
          lr=pars.parse_args().lr,
          epochs=pars.parse_args().e,
          early_stopping=pars.parse_args().es,
          path_weight=pars.parse_args().o,
          show_results=pars.parse_args().p)
