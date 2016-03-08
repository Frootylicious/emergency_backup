#! /usr/bin/env python3
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


    def _calculate_single_EC(self, combination_dict,
        save_path='results/emergency_capacities/', quantile=99):
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




    def plot_thing(self, f='s', c='c', b=1.00):
        alpha_list = []
        gamma_list = []
        self._get_chosen_combinations(f=f, c=c, b=b)
        self._calculate_all_EC()
        for combination in self.chosen_combinations:
            if combination['a'] not in alpha_list:
                alpha_list.append(combination['a'])
            if combination['g'] not in gamma_list:
                gamma_list.append(combination['g'])
        da = np.diff(alpha_list)[0]
        dg = np.diff(gamma_list)[0]
        EC_matrix = np.zeros((len(alpha_list), len(gamma_list)))
        load_str = 'results/emergency_capacities/'
        load_str += 'EC_' + self.file_string
        sum_loads = np.mean([np.sum(x) for x in self.loads])
        for i, a in enumerate(alpha_list):
            for j, g in enumerate(gamma_list):
                EC = np.load(load_str.format(**{'f':'s', 
                                                             'c':'c',
                                                             'b':b, 
                                                             'a':a, 
                                                             'g':g}))['arr_0']
                EC_matrix[i, j] = np.sum(EC[:,0]) / sum_loads
        a, g = np.mgrid[slice(min(alpha_list), max(alpha_list) + 2*da, da),
                        slice(min(gamma_list), max(gamma_list) + 2*dg, dg)]
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        plt.pcolormesh(a, g, EC_matrix, cmap='viridis')
        plt.title(r'$\frac{\mathcal{K}^{EB}_{EU}}{\left\langle\ L_{EU} \right\rangle}$', fontsize = 20)
#         plt.title(r'$\frac{\sum_n\ \mathcal{K}^E_n}{\left\langle\sum_n\ L_n\right\rangle}$', fontsize = 20)
        plt.xlabel(r'$\alpha$', fontsize = 20)
        plt.ylabel(r'$\gamma$', fontsize = 20)
        plt.axis([a.min(), a.max(), g.min(), g.max()])
        plt.yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
        plt.colorbar()
#         plt.show()
        if not os.path.exists('results/figures/'):
            os.mkdir('results/figures/')
        plt.savefig('results/figures/lol.png')
        return


## LAV ALT HERFRA IGEN MED DE NYE FEATURES. ------------------------------------
    # -- Public methods --
    def get_caps(self, country, save_path='results/'):

        # Finds the difference between two consecutive alpha and gamma values
        a, g = self.agbcl_list[0]
        for pair in self.agbcl_list:
            if pair[0] != a:
                a_diff = pair[0]
                break
        for pair in self.agbcl_list:
            if pair[1] != g:
                g_diff = pair[1]
                break

        # Setting up alpha and gamma values for use with np.pcolormesh()
        a_list = [a for (a, g) in self.agbcl_list]
        g_list = [g for (a, g) in self.agbcl_list]
        a, g = np.mgrid[slice(min(a_list), max(a_list) + 2*a_diff, a_diff),
                slice(min(g_list), max(g_list) + 2*g_diff, g_diff)]

        # Get the backup capacities for the country
        print('Calculating emergency backup capacities...')
        print(country)
        caps = self._find_caps(country, save_path)

        return a, g, caps

    def get_caps_europe(self, save_path='results/'):
        # Finds the difference between two consecutive alpha and gamma values
        a, g = self.agbcl_list[0]
        for pair in self.agbcl_list:
            if pair[0] != a:
                a_diff = pair[0]
                break
        for pair in self.agbcl_list:
            if pair[1] != g:
                g_diff = pair[1]
                break

        # Setting up alpha and gamma values for use with np.pcolormesh()
        a_list = [a for (a, g) in self.agbcl_list]
        g_list = [g for (a, g) in self.agbcl_list]
        a, g = np.mgrid[slice(min(a_list), max(a_list) + 2*a_diff, a_diff),
                slice(min(g_list), max(g_list) + 2*g_diff, g_diff)]

        # Get the backup capacities for the countries
        print('Calculating emergency backup capacities...')
        caps = np.zeros((len(self.alpha_values), len(self.gamma_values)))
        for country in self.countries:
            print(country)
            caps += self._find_caps(country, save_path)

        # Get the loads sum
        loadSum = np.zeros(len(self.loads[21]))
        for l in xrange(len(self.loads)):
            print l
            loadSum += self.loads[l]

        return a, g, caps, loadSum


    def plot_caps(self, country, save_path='results/'):

        a, g, caps = self.get_caps(country, save_path)

        # PLOT ALL THE THINGS
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        plt.pcolormesh(a, g, caps/np.mean(self.loads[self.country_dict[country]]))
        plt.title(r'$%s\ \frac{\mathcal{K}^E}{\left\langle L\right\rangle}$' % country, fontsize = 20)
        plt.xlabel(r'$\alpha$', fontsize = 20)
        plt.ylabel(r'$\gamma$', fontsize = 20)
        plt.axis([a.min(), a.max(), g.min(), g.max()])
        plt.yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
        plt.colorbar()
        plt.show()

    def plot_caps_europe(self, save_path='results/'):

        a, g, caps, loadSum = self.get_caps_europe(country, save_path)

        # PLOT ALL THE THINGS
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        plt.pcolormesh(a, g, caps/np.mean(loadSum))
        plt.title(r'$\frac{\sum_n\ \mathcal{K}^E_n}{\left\langle\sum_n\ L_n\right\rangle}$', fontsize = 20)
        plt.xlabel(r'$\alpha$', fontsize = 20)
        plt.ylabel(r'$\gamma$', fontsize = 20)
        plt.axis([a.min(), a.max(), g.min(), g.max()])
        plt.yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
        plt.colorbar()
        plt.show()


    def get_avg_backups(self, country):
        # Finds the difference between two consecutive alpha and gamma values
        a, g = self.agbcl_list[0]
        for pair in self.agbcl_list:
            if pair[0] != a:
                a_diff = pair[0]
                break
        for pair in self.agbcl_list:
            if pair[1] != g:
                g_diff = pair[1]
                break

        # Setting up alpha and gamma values for use with np.pcolormesh()
        a_list = [a for (a, g) in self.agbcl_list]
        g_list = [g for (a, g) in self.agbcl_list]
        a, g = np.mgrid[slice(min(a_list), max(a_list) + 2*a_diff, a_diff),
                slice(min(g_list), max(g_list) + 2*g_diff, g_diff)]

        avg_backups = np.zeros((len(self.alpha_values), len(self.gamma_values)))
        for index, (a, g) in enumerate(self.agbcl_list):
            ia, ig = divmod(index, len(self.alpha_values))
            avg_backups[ia, ig] = self._avg_backup(country, a, g)

        return a, g, avg_backups

    def plot_avg_backups(self, country):

        a, g, avg_backups = self.get_avg_backups(country)

        # PLOT ALL THE THINGS
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        plt.pcolormesh(a, g, avg_backups)
        plt.title(r'%s\ $\frac{\mathcal{K}^B_n}{\left\langleL_n\right\rangle}$' % country, fontsize = 20)
        plt.xlabel(r'$\alpha$', fontsize = 20)
        plt.ylabel(r'$\gamma$', fontsize = 20)
        plt.axis([a.min(), a.max(), g.min(), g.max()])
        plt.yticks(np.arange(g.min(), g.max(), np.diff(g)[0, 0]))
        plt.colorbar()
        plt.show()









if __name__ == '__main__':
    #     iset = r'/home/simon/Dropbox/Root/Data/ISET/'
    B = BackupEurope('results/balancing/', 'data/')
#     B._find_caps('DK')
#     B.plot_avg_backups('DK')

