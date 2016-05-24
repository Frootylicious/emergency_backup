#! /usr/bin/env python
import settings.settings as s
import numpy as np
from solve_single_network import Data
from itertools import product
from tqdm import tqdm
import multiprocessing as mp
import os


class solve_multiple_networks():

    def __init__(self, alpha_list=[0.8], gamma_list=[1.0], beta_list=[1.0],
                 constrained=True, DC=True, mode='square'):
        self.alpha_list = np.array(alpha_list)
        self.gamma_list = np.array(gamma_list)
        self.beta_list = np.array(beta_list)
        self.constrained = constrained
        self.DC = DC
        self.mode = mode
        self.product_variables = product(self.alpha_list,
                                         self.gamma_list,
                                         self.beta_list,
                                         [self.constrained],
                                         [self.DC],
                                         [self.mode])
        for values in tqdm(self.product_variables):
            a = values[0]
            g = values[1]
            b = values[2]
            constrained = values[3]
            DC = values[4]
            mode = values[5]
            data = Data(a=a, g=g, b=b, mode=mode, constrained=constrained, DC=DC, save_F=True)


# class BalancingCalculation():
#     '''
#     Class to hold the balancing timeseries.
#     It needs a list of alphas, gammas and betas to iterate through.
#     '''
# 
#     def __init__(self, alpha_list=[0.8], gamma_list=[1.0], beta_list=[1.0],
#                  constrained=False, DC=False, mode='square'):
#         self.alpha_list = alpha_list
#         self.gamma_list = gamma_list
#         self.beta_list = beta_list
#         self.constrained = constrained
#         self.DC = DC
#         self.mode = mode
# 
#     def run(self):
#         '''
#         Runs multiprocessing on number of cores - 1.
#         '''
# #         cores = mp.cpu_count() - 2
#         cores = 4
#         pool = mp.Pool(cores)
#         s = 'Running multiprocessing with {0} jobs on {1} cores.'
#         print(s.format(len(self.alpha_list) * len(self.gamma_list) * len(beta_list), cores))
#         pool.imap_unordered(get_B, product(self.alpha_list,
#                                 self.gamma_list,
#                                 self.beta_list,
#                                 [self.constrained],
#                                 [self.DC],
#                                 [self.mode]))
#         # pool.map(get_B, product(self.alpha_list,
#         #                         self.gamma_list,
#         #                         self.beta_list,
#         #                         [self.constrained],
#         #                         [self.DC],
#         #                         [self.mode]))
#         pool.close()
#         pool.join()
# 
# # if __name__ == '__main__':
# #     gamma_list = [1.0]
# #     alpha_list = [0.00]
# #     beta_list = [0.00, np.inf]
# #     lol = BalancingCalculation(alpha_list=alpha_list,
# #                                gamma_list=gamma_list,
# #                                beta_list=beta_list,
# #                                constrained=True,
# #                                DC=True,
# #                                mode='square')
# #     lol.run()
