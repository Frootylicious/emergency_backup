 # -*- coding: utf8 -*-
import numpy as np
from data_solving import Data
from itertools import product
import multiprocessing as mp
import os
import time


## Initialize
# alpha_list = np.linspace(0.0, 1.0, 2)
# gamma_list = np.linspace(0.0, 2.0, 2)
alpha_list = np.array([0.8])
gamma_list = np.array([1.0])
beta_list = np.array([1.0])


def get_B(values):
    print "ALPHA = {0} - GAMMA = {1}".format(values[0], values[1])
    data = Data(solve=True,
                alpha=values[0],
                gamma=values[1],
                mode='copper square',
                DC=True,
                filename='emergency_temp')
    balancing_timeseries = np.zeros((len(data.M), data.M[0].nhours))
    for i, n in enumerate(data.M):
        balancing_timeseries[i, :] = n.balancing
    if not os.path.exists('results/balancing'):
        os.makedirs('results/balancing')
    np.savez_compressed('results/balancing/{0:.1f}_{1:.1f}.npz'.format(values[0], values[1]), balancing_timeseries)
    print "Saved balancing to file: 'results/balancing/{0:.1f}_{1:.1f}.npz'".format(values[0], values[1])
    return

cores = mp.cpu_count()
pool = mp.Pool(cores-2)
pool.map(get_B, product(alpha_list, gamma_list))
pool.close()
pool.join()
