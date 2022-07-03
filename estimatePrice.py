import argparse
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from train_model import load_model
from train_model import StandartScalerReverse, MinMaxScalerReverse
from train_model import StandartScaler, MinMaxScaler


def estimate_price(mileage):
    model_info = load_model('weights.txt')
    price = model_info['w0'] + model_info['w1'] * MinMaxScaler(mileage,
                                                               x_min=model_info['x_min'],
                                                               x_max=model_info['x_max'])
    return round(price)


if __name__ == '__main__':
    pars = argparse.ArgumentParser()
    pars.add_argument('mileage', type=int or float,  help='mileage')
    print(estimate_price(pars.parse_args().mileage))