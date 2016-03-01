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
    def __init__(self, path='results/balancing', ISET_path='data/', constr='u', flowscheme='s'):
        "docstring"
        self.path = path
        self.ISET_path = ISET_path
        self.combinations = self._read_from_file()
        self.flowscheme = flowscheme
        self.constr = constr
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
        Returns a list of the values present in the files in the form:
        [(c, f, a, g, b)]
        '''
        filename_list = os.listdir(self.path)
        combinations = []
        for name in filename_list:
            combinations.append({'c':name[0],
                                 'f':name[2],
                                 'a':float(name[5:9]),
                                 'g':float(name[11:15]),
                                 'b':float(name[17:21])})
        return combinations


    def _get_chosen_combinations(self, **kwargs):
        '''
        Function that extracts the wanted files in the self.combinations list.
        For instance all the files in the synchronized flow scheme can be found
        by:
            self.get_chosen_combinations(f='s').

        All unconstrained files with a gamma = 1.00 can be found by:
            self.get_chosen_combinations(c='u', g=1.00)

        returns a list of dictionaries with the desired values.
        '''
        def check_in_dict(dic, kwargs):
            for (name, value) in kwargs.items():
                if not dic[name] == value:
                    return False
            return True
        chosen_combinations = []
        for combination in self.combinations:
            if check_in_dict(combination, kwargs):
                chosen_combinations.append(combination)
        return chosen_combinations


#     def _get_agb_values(self):
#         '''
#         Get lists with the given alpha, gamma and beta values with the chosen
#         constraint and flow scheme.
#         '''
#         a = []
#         g = []
#         b = []
#         for tuple in self.combinations:
#             if tuple[0] == self.constr and tuple[1] == self.flowscheme:
#                 if not tuple[2] in a:
#                     a.append(tuple[2])
#                 if not tuple[3] in g:
#                     g.append(tuple[3])
#                 if not tuple[4] in b:
#                     b.append(tuple[4])
#         return a, g, b


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





    def _calculate_caps(self, country,
            save_path='results/emergency_capacities'):
        if not os.path.exists(save_path):
            os.mkdirs(save_path)
        country = self.country_dict[country]
        load_str = '{0}{1}_{2}_a{3:.2f}_g{4:.2f}_b{5:.2f}.npz'
        for index, (c, f, a, g, b) in enumerate(self.combinations):
            backup = np.load(load_str.format(self.path, c, f, a, g, b))['arr_0'][country]







    def _find_caps(self, country, save_path='results/'):
        """
        Finds the emergency capacities for a country for every alpha gamma pair and arranges them
        in an array for use with np.pcolormesh()

        For each country the emergency capacities are saved
        to file save_path/country_alpha_gamma_caps.npz
        """
        # Make sure the right folder exists. If not - create it.
        if not os.path.exists(save_path + 'emergency_caps/'):
            os.makedirs(save_path + 'emergency_caps/')
        caps = -np.ones((len(self.alpha_values), len(self.gamma_values)))
        country = self.country_dict[country]
        for index, (c, f, a, g, b) in enumerate(self.combinations):
            sys.stdout.write('alpha = %.2f, gamma = %.2f\r' % ( a, g))
            sys.stdout.flush()
            #print('alpha = %.2f, gamma = %.2f' % ( a, g))
            print self.alpha_values
            ia, ig = divmod(index, len(self.alpha_values))
            backup = np.load('%s%s_%s_a%.2f_g%.2f_b%.2f.npz'
                    % (self.path, c, f, a, g, b))['arr_0'][country]            
            caps[ia, ig], storage = self._storage_needs(backup, 99)
            np.savez_compressed('%s%s_%s_a%.2f_g%.2f_b%.2f_%s_caps.npz'
                    % (save_path + 'emergency_caps/',
                        c, f, a, g, b, self.countries[country]),
                    caps = caps[ia, ig])
            return caps

    def _avg_backup(self, country, alpha, gamma, beta):
        country = self.country_dict[country]
        load_str = '{0}{1}_{2}_a{3:.2f}_g{4:.2f}_b{5:.2f}.npz'
        backup = np.load(load_str.format(self.path,
            self.constr,
            self.flowscheme,
            alpha,
            gamma,
            beta))['arr_0'][country]
        return np.mean(backup)


#     def _avg_backup(self, country, alpha, gamma):
#         sys.stdout.write('alpha = %.2f, gamma = %.2f\r' % (alpha, gamma))
#         sys.stdout.flush()
#         country = self.country_dict[country]
#         backup = np.load('%s%.2f_%.2f.npz' % (self.path, alpha, gamma))['arr_0'][country]
#         return np.mean(backup)


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
    iset = 'data/'
    B = BackupEurope('results/balancing/', iset, constr='c', flowscheme='s')
#     B._find_caps('DK')
#     B.plot_avg_backups('DK')

