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
'''


class BackupEurope(object):
    """ Backup docstring"""
    def __init__(self, path, ISET_path):
        "docstring"
        self.path = path
        self.ISET_path = ISET_path
        (self.alpha_values, 
         self.gamma_values, 
         self.beta_values, 
         self.c_and_u, 
         self.l_and_s, 
         agbcl) = self._read_numbers_from_files()
        self.agbcl_list = sorted(agbcl, key=lambda agbcl: (agbcl[0], 
                                                           agbcl[1],
                                                           agbcl[2], 
                                                           agbcl[3], 
                                                           agbcl[4]))
        self.countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                          'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                          'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
        self.country_dict = dict(zip(self.countries, list(range(len(self.countries)))))
        self.loads = [np.load('%sISET_country_%s.npz'\
                              % (self.ISET_path, self.countries[node]))['L']\
                      for node in range(len(self.countries))]

    # -- Private methods --
    def _read_numbers_from_files(self):
        """
        File names follow the convention:
        A_B_a.aa_g.gg_b.bb.npz
        A is either 'c' or 'u' for constrained/unconstrained
        B is either 'l' or 's' for linear/square (localized/synchronized flow scheme)
        a.aa is the alfa value
        g.gg is the gamma value
        b.bb is the beta value


        Return
        ------
        A list of touples containing each combination of (alpha, gamma, beta) in the filenames
        """
        filename_list = os.listdir(self.path)
        alpha_values = []
        gamma_values = []
        beta_values = []
        c_u = []
        l_s= []
        for name in filename_list:
            c = 'c' in name
            u = 'u' in name
            l = 'l' in name
            s = 's' in name
            a_index = name.find('a')
            g_index = name.find('g')
            b_index = name.find('b')
            a = float(name[a_index + 1: a_index + 5])
            g = float(name[g_index + 1: g_index + 5])
            b = float(name[b_index + 1: b_index + 5])
            if a not in alpha_values:
                alpha_values.append(a)
            if g not in gamma_values:
                gamma_values.append(g)
            if b not in beta_values:
                beta_values.append(g)
            if c and 'c' not in c_u:
                c_u.append('c')
            elif u and 'u' not in c_u:
                c_u.append('u')
            if l and 'l' not in l_s:
                l_s.append('l')
            elif s and 's' not in l_s:
                l_s.append('s')

        return (alpha_values, 
                gamma_values, 
                beta_values,
                c_u,
                l_s,
                list(product(alpha_values, 
                             gamma_values, 
                             beta_values,
                             c_u,
                             l_s)))

    def _quantile(self, quantile, dataset, cutzeros=True):
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

    def _storage_needs(self, backup, quantile):
        storage = np.zeros(len(backup))
        for index, val in enumerate(backup):
            if val >= quantile:
                storage[index] = storage[index] - (val - quantile)
            else:
                storage[index] = storage[index] + (quantile - val)
                if storage[index] > 0:
                    storage[index] = 0
        return -min(storage), storage

    def _find_caps(self, country, save_path='results/'):
        """
        Finds the emergency capacities for a country for every alpha gamma pair and arranges them
        in an array for use with np.pcolormesh()
        
        For each country the emergency capacities are saved
        to file save_path/country_alpha_gamma_caps.npz
        """
        caps = np.zeros((len(self.alpha_values), len(self.gamma_values)))
        country = self.country_dict[country]
        for index, (a, g) in enumerate(self.agbcl_list):
            sys.stdout.write('alpha = %.2f, gamma = %.2f\r' % ( a, g))
            sys.stdout.flush()
            #print('alpha = %.2f, gamma = %.2f' % ( a, g))
            ia, ig = divmod(index, len(self.alpha_values))
            backup = np.load('%s%.2f_%.2f.npz' % (self.path, a, g))['arr_0'][country]
            q = self._quantile(99, backup, cutzeros=True)
            caps[ia, ig], storage = self._storage_needs(backup, q)
            np.savez_compressed('%s%s_%.2f_%.2f_caps.npz' % (save_path, self.countries[country], a, g)
                                , caps = caps[ia, ig])
        return caps

    def _avg_backup(self, country, alpha, gamma):
        sys.stdout.write('alpha = %.2f, gamma = %.2f\r' % (alpha, gamma))
        sys.stdout.flush()
        country = self.country_dict[country]
        backup = np.load('%s%.2f_%.2f.npz' % (self.path, alpha, gamma))['arr_0'][country]
        return np.mean(backup)
        

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
    iset = 'data/'
    B = BackupEurope('results/balancing/', iset)
#     B.plot_avg_backups('DK')

