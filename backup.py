#! /usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import os, os.path
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import colormaps as cmaps

import sys

'''
TODO:
    The different a/g/b are stored correctly. Now we need to find out how to
    represent the data with constrained/unconstrained.

    We probably need to be able to choose what kind of attributes, we want to
    look at (constrained/unconstrained/linear/square).

    Blocks that are out-commented should now be obsolete.
'''


class BackupEurope(object):
    """ Backup docstring"""
    def __init__(self, path='results/balancing', ISET_path='data/'):
        "docstring"
        self.path = path
        self.ISET_path = ISET_path
        # Saving all combinations present from files.
        self.all_combinations = self._read_from_file()
        self.chosen_combinations = self._get_chosen_combinations()
        self.file_string = '{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npz'
        self.countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                          'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                          'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
        self.country_dict = dict(zip(self.countries, list(range(len(self.countries)))))
        self.loads = [np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['L']\
                for node in range(len(self.countries))]

        # -- Private methods --
    def _read_from_file(self):
        '''
        Returns a list of dictionaries with the values present in the files in the form:
            [{'c':'u', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00},
             {'c':'c', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00}]
        '''
        filename_list = os.listdir(self.path)
        all_combinations = []
        for name in filename_list:
            all_combinations.append({'c':name[0],
                                     'f':name[2],
                                     'a':float(name[5:9]),
                                     'g':float(name[11:15]),
                                     'b':float(name[17:21])})
        return all_combinations


    def _get_chosen_combinations(self, **kwargs):
        '''
        Function that extracts the wanted files in the self.all_combinations list.
        For instance all the files in the synchronized flow scheme can be found
        by:
            self.get_chosen_combinations(f='s').

        All unconstrained files with a gamma = 1.00 can be found by:
            self.get_chosen_combinations(c='u', g=1.00)

        returns a list of dictionaries with the desired values.
        For instance:
            [{'c':'u', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00},
             {'c':'c', 'f':'s', 'a':0.80, 'g':1.00, 'b':0.50}]
        '''
        def _check_in_dict(dic, kwargs):
            # Returns false if one of the kwargs supplied differ from the dic.
            for (name, value) in kwargs.items():
                if not dic[name] == value:
                    return False
            return True
        chosen_combinations = []
        for combination in self.all_combinations:
            if _check_in_dict(combination, kwargs):
                chosen_combinations.append(combination)
        if len(chosen_combinations) == 0:
            raise ValueError('No files with {0} found!'.format(kwargs))
        self.chosen_combinations = chosen_combinations
        return chosen_combinations


    def _quantile(self, quantile, dataset, cutzeros=False):
        """
        Takes a list of numbers, converts it to a list without zeros
        and returns the value of the 99% quantile.
        """
        if cutzeros:
            # Removing zeros
            dataset = dataset[np.nonzero(dataset)]
        # Convert to numpy histogram
        hist, bin_edges = np.histogram(dataset, bins = 10000, normed = True)
        dif = np.diff(bin_edges)[0]
        q = 0
        for index, val in enumerate(reversed(hist)):
            q += val*dif
            if q > 1 - float(quantile)/100:
                #print 'Found %.3f quantile' % (1 - q)
                return bin_edges[-index]


    def _storage_needs(self, backup_timeseries, quantile):
        """
        Arguments
        ---------
        backup:
        A timeseries of backups for a given node in the network

        quantile:
        Eg. 99% quantile
        """
        storage = np.zeros(len(backup_timeseries))
        q = self._quantile(quantile, backup_timeseries)
        for index, val in enumerate(backup_timeseries):
            if val >= q:
                storage[index] = storage[index] - (val - q)
            else:
                storage[index] = storage[index] + (q - val)
                if storage[index] > 0:
                    storage[index] = 0
        return -min(storage), storage



    def _calculate_all_EC(self, save_path='results/emergency_capacities/', quantile=99):
        '''
        Function that calculates emergency storage capacities for all countries
        in the files given by self.chosen_combinations and saves them to files.

        The saved file is a 30*2 numpy array with the emergency backup capacity
        in row 0 and the average backup in row 2.

        Saves in the form:
        EC_{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npy

        '''
        # Looping over all countries
        for index, combination in enumerate(self.chosen_combinations):
            self._calculate_single_EC(combination)
        return


    def _calculate_single_EC(self, 
                             combination_dict,
                             save_path='results/emergency_capacities/', 
                             quantile=99):
        # Check if path exists. Create it if not.
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if os.path.isfile(save_path + 'EC_' +
                self.file_string.format(**combination_dict)):
            print('EC-file {0} already exists - skipping.'.format(combination_dict))
        else:
            combination_caps = np.zeros((len(self.countries), 2))
            backup = np.load(self.path +
                    self.file_string.format(**combination_dict))['arr_0']
            for i, country_backup in enumerate(backup):
                combination_caps[i, 0] = self._storage_needs(country_backup,
                        quantile)[0]
                combination_caps[i, 1] = np.mean(country_backup)
            np.savez(save_path + 'EC_' +
                    self.file_string.format(**combination_dict), combination_caps)
            print('Saved EC-file: {0}'.format(combination_dict))
        return




    def plot_colormap(self, f='s', c='c', b=0.50):
        # Preparing the lists for alpha and gamma values.
        alpha_list = []
        gamma_list = []
        # Only get the combinations with the wanted f and c.
        self._get_chosen_combinations(f=f, c=c, b=b)
        # Calculate the emergency capacities for the wanted values.
        self._calculate_all_EC()
        # This loop appends all values of alpha and gamma into the prepared
        # lists.
        for combination in self.chosen_combinations:
            if combination['a'] not in alpha_list:
                alpha_list.append(combination['a'])
            if combination['g'] not in gamma_list:
                gamma_list.append(combination['g'])
        # We have to sort the lists to make the correct grids.
        alpha_list.sort()
        gamma_list.sort()
        # Finding the space between first and second value in the lists.
        da = float('{0:.2f}'.format(np.diff(alpha_list)[0]))
        dg = float('{0:.2f}'.format(np.diff(gamma_list)[0]))
        EC_matrix = np.zeros((len(alpha_list), len(gamma_list)))
        load_str = 'results/emergency_capacities/'
        load_str += 'EC_' + self.file_string
        # Calculating the mean of the sum of the loads for europe.
        mean_sum_loads = np.mean([np.sum(x) for x in self.loads])
        for i, a in enumerate(alpha_list):
            for j, g in enumerate(gamma_list):
                load_dict = {'f':f, 'c':c, 'b':b, 'a':a, 'g':g}
                EC = np.load(load_str.format(**load_dict))['arr_0']
                EC_matrix[i, j] = np.sum(EC[:,0]) / mean_sum_loads
        # Creating the plot
        a, g = np.mgrid[slice(min(alpha_list), max(alpha_list) + 2 * da, da),
                        slice(min(gamma_list), max(gamma_list) + 2 * dg, dg)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        cms = ax.pcolormesh(a, g, EC_matrix, cmap='viridis')
        str1 = r'$\frac{\mathcal{K}^{EB}_{EU}}{\left\langle L_{EU}\right\rangle}$'
        str2 = 'constrained' if c=='c' else 'unconstrained'
        str3 = 'synchronized' if f=='s' else 'localized'
        str4 = str1 + ' with ' + str3 + ' ' + str2 + r' flow $\beta={0}$'.format(b)
        ax.set_title(str4, y=1.08, fontsize=15)
        ax.set_xlabel(r'$\alpha$', fontsize = 20)
        ax.set_ylabel(r'$\gamma$', fontsize = 20)
        ax.set_xlim([a.min(), a.max()])
        ax.set_ylim([g.min(), g.max()])
        ax.set_yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
        fig.colorbar(cms)
        plt.tight_layout()
        if not os.path.exists('results/figures/'):
            os.mkdir('results/figures/')
        save_str = 'colormap_b{0:.2f}.png'.format(b)
        plt.savefig('results/figures/' + save_str)
        plt.close()
        return


if __name__ == '__main__':
    B = BackupEurope('results/balancing/', 'data/')
    B.plot_colormap()
#     B._find_caps('DK')
#     B.plot_avg_backups('DK')

