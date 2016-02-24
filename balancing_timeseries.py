 # -*- coding: utf8 -*-
import numpy as np
from data_solving import Data
from itertools import product
import multiprocessing as mp
import os
import time


'''
TODO:
    Fix multiprocessing (broken atm.)
    Figure out if a class is really needed.
'''

class  BalancingCalculation():
    '''
    Class to hold the balancing timeseries.
    '''
    def __init__(self, alpha_list=[0.8], gamma_list=[1.0], beta_list=[1.0]):
        self.alpha_list = alpha_list
        self.gamma_list = gamma_list
        self.beta_list = beta_list

    def run_multiprocessing(self):
        '''
        Runs multiprocessing on number of cores - 2.
        '''

        def get_B(values):
            print "ALPHA = {0}\nGAMMA = {1}\n BETA = {2}".format(values[0],
                                                                 values[1],
                                                                 values[2])
            # Set up the Nodes-object
            data = Data(solve=True,
                        alpha=values[0],
                        gamma=values[1],
                        mode='copper square',
                        DC=True,
                        filename='emergency_temp')
            # Initializing the variable.
            balancing_timeseries = np.zeros((len(data.M), data.M[0].nhours))
            # Change the variable in place.
            for i, n in enumerate(data.M):
                balancing_timeseries[i, :] = n.balancing
            # Make sure the right folder exists. If not - create it.
            if not os.path.exists('results/balancing'):
                os.makedirs('results/balancing')
            # Save variable.
            np.savez_compressed('results/balancing/{0:.2f}_{1:.2f}_{2:.2f}.npz'.format(values[0],
                                                                                    values[1],
                                                                                    values[2]),
                                                                                    balancing_timeseries)

            print "Saved balancing to file: 'results/balancing/{0:.1f}_{1:.1f}.npz'".format(values[0],
                                                                                            values[1])

        cores = mp.cpu_count()
        pool = mp.Pool(cores - 2)
        pool.map(get_B, product(self.alpha_list, self.gamma_list, self.beta_list))
        pool.close()
        pool.join()
