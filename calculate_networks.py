#! /usr/bin/env python
import settings as s
import numpy as np
from data_solving import Data
from itertools import product
import multiprocessing as mp
from tqdm import tqdm
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
        3: constrained [True/False]
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

    # Checking if the file already exists. In this case - skip it.
    f = s.nodes_fullname.format(c=str_constrained, f=str_lin_syn, a=a, g=g, b=b)
    if os.path.isfile(f):
        print('file: "{0}" already exists - skipping.'.format(f))
        return
    else:
        data = Data(a=a, g=g, b=b, mode=mode, constrained=constrained, DC=DC, save_F=True)


class BalancingCalculation():
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
#         cores = mp.cpu_count() - 2
        cores = 4
        pool = mp.Pool(cores)
        s = 'Running multiprocessing with {0} jobs on {1} cores.'
        print(s.format(len(self.alpha_list) * len(self.gamma_list) * len(beta_list), cores))
        tqdm(pool.imap_unordered(get_B, product(self.alpha_list,
                                self.gamma_list,
                                self.beta_list,
                                [self.constrained],
                                [self.DC],
                                [self.mode])))
        # pool.map(get_B, product(self.alpha_list,
        #                         self.gamma_list,
        #                         self.beta_list,
        #                         [self.constrained],
        #                         [self.DC],
        #                         [self.mode]))
        pool.close()
        pool.join()

if __name__ == '__main__':
    #     alpha_list = np.linspace(0, 1, 11)
    #     gamma_list = np.linspace(0, 2, 11)
    beta_list = np.linspace(0, 1.5, 16)
    #     beta_list = [0.50, 0.75, 1.00]
    alpha_list = [0.8]
#     gamma_list = [1.0]
    # beta_list = [1.0]
#     alpha_list = np.linspace(0, 1, 51)
    gamma_list = [1.0]
#     beta_list = [np.inf]
    lol = BalancingCalculation(alpha_list=alpha_list,
                               gamma_list=gamma_list,
                               beta_list=beta_list,
                               constrained=True,
                               DC=True,
                               mode='square')
    lol.run()
