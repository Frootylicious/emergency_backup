#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
# matplotlib.use('Agg')
import sys
import os, os.path
import numpy as np
import colormaps as cmaps
import matplotlib.pyplot as plt
import regions.classes as cl
from itertools import product
from data_solving import Data


'''
TODO:
'''


class BackupEurope(object):
    """ Backup docstring"""
    def __init__(self, N_path='results/N/', ISET_path='data/'):
        "docstring"
        self.N_path = N_path
        self.ISET_path = ISET_path
        # Saving all combinations present from files.
        self.all_combinations = self._read_from_file()
        self.chosen_combinations = self.get_chosen_combinations()
        self.N_str = '{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}'
        self.countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                          'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                          'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
        self.country_dict = dict(zip(self.countries, list(range(len(self.countries)))))
        if not os.path.exists('results/figures/'):
            os.mkdir('results/figures/')
        self.loads = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['L']\
                for node in range(len(self.countries))])

    def _read_from_file(self):
        '''
        Returns a list of dictionaries with the values present in the files in the form:
            [{'c':'u', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00},
             {'c':'c', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00}]
        '''
        filename_list = os.listdir(self.N_path)
        all_combinations = []
        for name in filename_list:
            all_combinations.append({'c':name[0],
                                     'f':name[2],
                                     'a':float(name[5:9]),
                                     'g':float(name[11:15]),
                                     'b':float(name[17:21])})
        return all_combinations


    def get_chosen_combinations(self, **kwargs):
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
                value = np.array(value)
                if not dic[name] in value:
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
        # Calculating all EC
        for combination in self.all_combinations:
            self._calculate_single_EC(combination)
        return

    def _calculate_chosen_EC(self, save_path='results/emergency_capacities/',  quantile=99):
        # Calculating chosen EC
        for combination in self.chosen_combinations:
            self._calculate_single_EC(combination)


    def _calculate_single_EC(self,
                             combination_dict,
                             save_path='results/emergency_capacities/',
                             quantile=99):
        '''
        Function that calculates emergency storage capacities for all countries
        in the file given by the combination-dictionary and saves them to files.

        The saved file is a 30 numpy array with the emergency backup capacity

        Saves in the form:
        EC_{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npy

        '''
        # Check if path exists. Create it if not.
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if os.path.isfile(save_path + 'EC_' +
                self.N_str.format(**combination_dict)):
            print('EC-file {0} already exists - skipping.'.format(combination_dict))
        else:
            combination_caps = np.zeros(len(self.countries))
            nodes = np.load(self.N_path + self.N_str.format(**combination_dict) + '_N.npz')
            balancing = nodes['balancing']
            for i, country_backup in enumerate(balancing):
                combination_caps[i] = self._storage_needs(country_backup, quantile)[0]
            np.savez(save_path + 'EC_' +
                    self.N_str.format(**combination_dict) + '.npz', combination_caps)
            print('Saved EC-file: {0}'.format(combination_dict))
        return


    def plot_colormap(self, f='s', c='c', b=1.00, a_amount=11, g_amount=11):
        # Preparing the lists for alpha and gamma values.
        alpha_list = np.linspace(0, 1, a_amount)
        gamma_list = np.linspace(0, 2, g_amount)
        # Only get the combinations with the wanted f and c.
        self.get_chosen_combinations(f=f, c=c, b=b, a=alpha_list, g=gamma_list)
        # Calculate the emergency capacities for the wanted values.
        self._calculate_chosen_EC()
        alpha_list.sort()
        gamma_list.sort()
        # Finding the space between first and second value in the lists.
        da = float('{0:.2f}'.format(np.diff(alpha_list)[0]))
        dg = float('{0:.2f}'.format(np.diff(gamma_list)[0]))
        EC_matrix = np.empty((len(alpha_list), len(gamma_list)))
        EC_matrix[:] = np.nan
        load_str = 'results/emergency_capacities/'
        load_str += 'EC_' + self.N_str + '.npz'
        # Calculating the mean of the sum of the loads for europe.
        mean_sum_loads = np.mean(np.sum(self.loads, axis=0)) * 1000
        number_of_nans = 0
        for i, a in enumerate(alpha_list):
            for j, g in enumerate(gamma_list):
                load_dict = {'f':f, 'c':c, 'b':b, 'a':a, 'g':g}
                if os.path.isfile(load_str.format(**load_dict)):
                    EC = np.load(load_str.format(**load_dict))['arr_0']
                    EC_matrix[i, j] = np.sum(EC) / mean_sum_loads
                else:
                    number_of_nans += 1
        print("{0} files didn't exist - set as NaN".format(number_of_nans))
        # Important to do this:
        EC_matrix = np.ma.masked_invalid(EC_matrix)
        # Creating the plot
        a, g = np.mgrid[slice(min(alpha_list), max(alpha_list) + 2 * da, da),
                        slice(min(gamma_list), max(gamma_list) + 2 * dg, dg)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        plt.set_cmap(cmaps.viridis)
        cmap = plt.get_cmap()
        cmap.set_bad(color='w', alpha=1.)
        cms = ax.pcolormesh(a, g, EC_matrix, cmap='viridis')
        # Prepare the strings.
        str1 = r'$\frac{\mathcal{K}^{EB}_{EU}}{\left\langle L_{EU}\right\rangle}$'
        str2 = 'constrained' if c=='c' else 'unconstrained'
        str3 = 'synchronized' if f=='s' else 'localized'
        str4 = str1 + ' with ' + str3 + ' ' + str2 + r' flow $\beta={0}$'.format(b)
        ax.set_title(str4, y=1.08, fontsize=15)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$\gamma$', fontsize=20)
        ax.set_xlim([a.min(), a.max()])
        ax.set_ylim([g.min(), g.max()])
        # Setting ticks in the middle of the squares.
        xticks = np.array(alpha_list) + 0.5 * da
        yticks = np.array(gamma_list) + 0.5 * dg
        xlabels = [str(x) for x in alpha_list]
        ylabels = [str(y) for y in gamma_list]
        ax.xaxis.set(ticks=xticks, ticklabels=xlabels)
        ax.yaxis.set(ticks=yticks, ticklabels=ylabels)
        fig.colorbar(cms)
        plt.tight_layout()
        # Checking if the folder exists - create it if it doesn't
        save_str = 'colormapAG_b{0:.2f}.png'.format(b)
        plt.savefig('results/figures/' + save_str)
        print('Saved file "{0}".'.format(save_str))
        plt.close()
        return

    def plot_timeseries_EU(self, a=0.80, g=1.00, b=1.00):
#         B = Data(solve=True, a=a, g=g, b=b, constrained=True, DC=True)
        countries = [i + '.npz' for i in self.countries]
        load_str = 'N/' + self.N_str.format(c='c', f='s', a=a, g=g, b=b) + '_N.npz'
        N = cl.Nodes(load_filename=load_str,
                     files=countries,
                     path='data/',
                     prefix='ISET_country_')

        timeseries = np.zeros((30, N[0].nhours))

        for i, n in enumerate(N):
            timeseries[i] = n.load - n.get_solar() - n.get_wind() - self._quantile(99,
                    n.get_balancing()) + n.get_export() - n.get_import() + n.get_curtailment()

        EUL_avg = np.mean(np.sum([x.load for x in N], axis=0))

        K_EB_str = 'results/emergency_capacities/EC_' + self.N_str.format(c='c',
                                                                          f='s',
                                                                          a=a,
                                                                          g=g,
                                                                          b=b) + '.npz'
        K_EB = np.load(K_EB_str)
        print(K_EB.f.arr_0)
        K_EB = sum(K_EB.f.arr_0) / EUL_avg

        timeseries_EU = np.sum(timeseries, axis=0) / EUL_avg
        timeseries_EU[timeseries_EU < 0] = 0
        self.timeseries_EU = timeseries_EU

        title_str1 = r'$\frac{L_{EU}-G_{EU}^R - K_{EU}^{B99}' 
        title_str2 = r' - I_{EU} + E_{EU}}{\left< L_{EU} \right>}$'
        txt_str1 = r'$\frac{{K_{{EU}}^{{EB}}}}{{\left<L_{{EU}}\right>}} = {0:.2f}$'
        txt_str2 = r'$\alpha = {0}, \gamma = {1}, \beta = {2}$'.format(a, g, b)

        fig, (ax)  = plt.subplots(1, 1, sharex=True)
        ax.plot(timeseries_EU)
        ax.set_title(title_str1 + title_str2, fontsize=20, y=0.9, x=0.3)
        fig.text(x=0.1, y=0.7, s=txt_str1.format(K_EB), fontsize=20)
        fig.text(x=0.1, y=0.6, s=txt_str2, fontsize=15)
        plt.tight_layout()
#         plt.show()
        plt.savefig('results/figures/timeseriesEU.png')
        plt.close()
        return


    def plot_alpha(self, gamma=1.00, beta=1.00, c='c', f='s'):
        self.get_chosen_combinations(g=gamma, b=beta, c=c, f=f)
        self._calculate_chosen_EC()
        alpha_list = []
        EC_list = []
        filepath = 'results/emergency_capacities/EC_'
        EUL = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['L']\
                for node in range(len(self.countries))])
        EUL = np.sum(EUL, axis=0) * 1000
        for combination in self.chosen_combinations:
            alpha_list.append(combination['a'])
            EC = np.load(filepath + self.N_str.format(**combination) + '.npz')
            EC = np.mean(np.sum(EC.f.arr_0, axis=0))
            EC_list.append(EC/np.mean(EUL))

        legend = r'$\frac{\mathcal{K}_{EU}^{EB}}{\left\langle L_{EU}\right\rangle}$'
        fig, (ax)  = plt.subplots(1, 1, sharex=True)
        ax.plot(alpha_list, EC_list, '.k', label=legend, ms=10)
        str2 = 'constrained' if c=='c' else 'unconstrained'
        str3 = 'synchronized' if f=='s' else 'localized'
        str4 = str3 + ' ' + str2 + r' flow $\beta={0}$'.format(beta)
        ax.set_title(str4, y=1.08, fontsize=15)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.legend(loc=2)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        plt.savefig('results/figures/EB_Alpha.png')
        plt.close()
        return

if __name__ == '__main__':
    B = BackupEurope()
#     B.plot_colormap()
#     B.plot_timeseries()
#     B.plot_timeseries_EU()
#     B.plot_alpha()


