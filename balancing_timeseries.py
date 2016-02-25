 # -*- coding: utf8 -*-
import numpy as np
from data_solving import Data
from itertools import product
import multiprocessing as mp
import os
import time


'''
TODO:
'''


def get_B(values):
    # Set up the Nodes-object
    data = Data(solve=True,
                a=values[0],
                g=values[1],
                b=values[2],
                constrained=values[3],
                DC=values[4],
                mode='square',
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
    print("Saved balancing to file: 'results/balancing/{0:.1f}_{1:.1f}.npz'".format(values[0],
                                                                                    values[1]))

class  BalancingCalculation():
    '''
    Class to hold the balancing timeseries.
    It needs a list of alphas, gammas and betas to iterate through.
    '''
    def __init__(self, alpha_list=[0.8], gamma_list=[1.0], beta_list=[1.0],
                 constrained=False, DC=False):
        self.alpha_list = alpha_list
        self.gamma_list = gamma_list
        self.beta_list = beta_list
        self.constrained = constrained
        self.DC = DC


    def run(self):
        '''
        Runs multiprocessing on number of cores - 2.
        '''
        cores = mp.cpu_count()
        pool = mp.Pool(cores - 2)
        s = 'Running multiprocessing with {0} jobs on {1} cores.'
        print(s.format(len(self.alpha_list) * len(self.gamma_list) * len(beta_list), cores))
        pool.map(get_B, product(self.alpha_list, self.gamma_list,
                                self.beta_list, [self.constrained], [self.DC]))
        pool.close()
        pool.join()

if __name__ == '__main__':
    beta_list = np.linspace(0, 1, 5)
    lol = BalancingCalculation(beta_list=beta_list)
    lol.run()
