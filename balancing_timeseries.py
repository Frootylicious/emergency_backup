#! /usr/bin/env python
import numpy as np
from data_solving import Data
from itertools import product
import multiprocessing as mp
import os


'''
TODO:
'''


def get_B(values):
    '''
    Function needed for the multiprocessing. 

    The variables in "values" are packed together to enable multiprocessing.
    values:
        0: alpha [float]
        1: gamma [float]
        2: beta [float]
        3: constraned [True/False]
        4: DC [True/False]
        5: mode [string] ('linear'/'square')
    '''
    # Set up the Nodes-object
    a = values[0]
    g = values[1]
    b = values[2]
    constrained = values[3]
    DC = values[4]
    mode = values[5]
    # Setting the naming. 'c' for constrained, 'u' for unconstrained.
    # 's' for synchronized/square and 'l' for localized/linear.
    str_constrained = 'c' if constrained else 'u'
    str_lin_syn = 's' if 'square' in mode else 'l'
    # The complete filename.
    filename = ('results/balancing/'
                '{0}_{1}_a{2:.2f}_g{3:.2f}_b{4:.2f}.npz').format(str_constrained,
                                                                 str_lin_syn,
                                                                 a, g, b)

    # Checking if the file already exists. In this case - skip it.
    if os.path.isfile(filename):
        print('file: "{0}" already exists - skipping.'.format(filename))
        return

    # Instantiating class Data. No save is needed, as we only want the
    # balancing.
    data = Data(solve=True,
                a=a,
                g=g,
                b=b,
                constrained=constrained,
                DC=DC,
                mode=mode,
                filename='emergency_storage_temp',
                save=False)
    # Initializing the variable.
    balancing_timeseries = np.zeros((len(data.M), data.M[0].nhours))
    # Change the variable in place.
    for i, n in enumerate(data.M):
        balancing_timeseries[i, :] = n.balancing
    # Make sure the right folder exists. If not - create it.
    if not os.path.exists('results/balancing'):
        os.makedirs('results/balancing')
    # Save variable.
    np.savez_compressed(filename, balancing_timeseries)
    print("Saved balancing to file: '{0}'".format(filename))


class  BalancingCalculation():
    '''
    Class to hold the balancing timeseries.
    It needs a list of alphas, gammas and betas to iterate through.
    '''
    def __init__(self, alpha_list=[0.8], gamma_list=[1.0], beta_list=[1.0],
                 constrained=False, DC=False, mode='square'):
        self.alpha_list = alpha_list
        self.gamma_list = gamma_list
        self.beta_list = beta_list
        self.constrained = constrained
        self.DC = DC
        self.mode = mode


    def run(self):
        '''
        Runs multiprocessing on number of cores - 1.
        '''
        cores = mp.cpu_count()
        pool = mp.Pool(cores - 1)
        s = 'Running multiprocessing with {0} jobs on {1} cores.'
        print(s.format(len(self.alpha_list) * len(self.gamma_list) * len(beta_list), cores))
        pool.map(get_B, product(self.alpha_list, 
                                self.gamma_list,
                                self.beta_list, 
                                [self.constrained],
                                [self.DC], 
                                [self.mode]))
        pool.close()
        pool.join()

if __name__ == '__main__':
#     alpha_list = np.linspace(0, 1, 11)
#     gamma_list = np.linspace(0, 2, 11)
#     beta_list = np.linspace(0, 1, 1)
    alpha_list = [0.8]
    gamma_list = [1.0]
    beta_list = [1.0]
    lol = BalancingCalculation(alpha_list = alpha_list,
                               gamma_list = gamma_list,
                               beta_list=beta_list,
                               constrained=True,
                               mode='square')
    lol.run()
