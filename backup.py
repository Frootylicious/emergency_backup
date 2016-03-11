#! /usr/bin/env python3
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import sys
import os, os.path
import numpy as np
import colormaps as cmaps
import matplotlib.pyplot as plt
from itertools import product


'''
TODO:
    Find out if the load and backups are in the right units.
'''


class BackupEurope(object):
    """ Backup docstring"""
    def __init__(self, path='results/balancing', ISET_path='data/'):
        "docstring"
        self.path = path
        self.ISET_path = ISET_path
        # Saving all combinations present from files.
        self.all_combinations = self._read_from_file()
        self.chosen_combinations = self.get_chosen_combinations()
        self.file_string = '{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npz'
        self.countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                          'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                          'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
        self.country_dict = dict(zip(self.countries, list(range(len(self.countries)))))
        self.loads = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['L']\
                for node in range(len(self.countries))])
        # Making loads MW instead of GW
#         self.loads *= 1000

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

        The saved file is a 30*2 numpy array with the emergency backup capacity
        in row 0 and the average backup in row 2.

        Saves in the form:
        EC_{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npy

        '''
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


    def plot_colormap(self, f='s', c='c', b=1.00, a_amount=3, g_amount=3):
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
        load_str += 'EC_' + self.file_string
        # Calculating the mean of the sum of the loads for europe.
        mean_sum_loads = np.mean(np.sum(self.loads, axis=0)) * 1000
        number_of_nans = 0
        for i, a in enumerate(alpha_list):
            for j, g in enumerate(gamma_list):
                load_dict = {'f':f, 'c':c, 'b':b, 'a':a, 'g':g}
                if os.path.isfile(load_str.format(**load_dict)):
                    EC = np.load(load_str.format(**load_dict))['arr_0']
                    EC_matrix[i, j] = np.sum(EC[:,0]) / mean_sum_loads
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
        if not os.path.exists('results/figures/'):
            os.mkdir('results/figures/')
        save_str = 'colormapAG_b{0:.2f}.png'.format(b)
        plt.savefig('results/figures/' + save_str)
        print('Saved file "{0}".'.format(save_str))
        plt.close()
        return

    def plot_timeseries_EU(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        filename = 'c_s_a0.80_g1.00_b1.00.npz'

        EUB = np.load('results/balancing/' + filename)
        EUB = np.sum(EUB.f.arr_0, axis=0)
        EU_EB = np.load('results/emergency_capacities/EC_' + filename)
        EU_EB = np.sum(EU_EB.f.arr_0, axis=0)[0]
        EU_Bq = self._quantile(99, EUB)
        print EU_EB
        EU = np.load('data/ISET_country_DE.npz')
        EUW = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['Gw']\
                for node in range(len(self.countries))])
        EUS = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['Gs']\
                for node in range(len(self.countries))])
        EUW = np.sum(EUW, axis=0)
        EUS = np.sum(EUS, axis=0)
        EUL = np.array([np.load('%sISET_country_%s.npz'\
                % (self.ISET_path, self.countries[node]))['L']\
                for node in range(len(self.countries))])
        EUL = np.sum(EUL, axis=0) * 1000
        EUG = EUW + EUS * 1000
        EUL_avg = np.mean(EUL)
        print('Mean Balancing: ', np.mean(EUB)/np.mean(EUL))
        print('Mean load: ', np.mean(EUL)/np.mean(EUL))
        print('Mean total generation ', np.mean(EUG)/np.mean(EUL))
        print('Emergency Backup capacity: ', EU_EB/np.mean(EUL))
#         lol = DKL - DKG - DKB
        lol = EUL - EUG - EU_Bq
        lol2 = EUB - EU_Bq
        lol[lol < 0] = 0

        days = 365*7

        ax1.plot(EUG[:days*24]/np.mean(EUL))
        ax2.plot(EUL[:days*24]/np.mean(EUL))
        ax3.plot(lol2[:days*24]/np.mean(EUL))
        ax4.plot(lol[:days*24]/np.mean(EUL))
        plt.savefig('results/figures/lol.png')
        return

    def plot_timeseries_country(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        filename = 'c_s_a0.80_g1.00_b1.00.npz'
        country = 18

        DKB = np.load('results/balancing/' + filename)
        DKB = DKB.f.arr_0[country]
        DK_EB = np.load('results/emergency_capacities/EC_' + filename)

        DK_EB = DK_EB.f.arr_0[country]
        DK_Bq = self._quantile(99, DKB)
        print(DK_Bq)
        DK = np.load('data/ISET_country_DE.npz')
        DKW = DK.f.Gw
        DKS = DK.f.Gs
        DKG = DKW + DKS * 1000
        DKL = DK.f.L * 1000
        DKL_avg = np.mean(DKL)
        print('Mean Balancing: ', np.mean(DKB))
        print('Mean load: ', np.mean(DKL))
        print('Mean total generation ', np.mean(DKG))
        print('Emergency Backup capacity: ', DK_EB)
#         lol = DKL - DKG - DKB
        lol = DKL - DKG - DK_Bq
        lol2 = DKB - DK_Bq
        lol[lol < 0] = 0

        days = 60

        ax1.plot(DKG[:days*24]/np.mean(DKL))
        ax2.plot(DKL[:days*24]/np.mean(DKL))
        ax3.plot(lol2[:days*24]/np.mean(DKL))
        ax4.plot(lol[:days*24]/np.mean(DKL))
        plt.savefig('results/figures/lol.png')
        return


    def plot_alpha(self, gamma=1.00, beta=1.00, c='c', f='s'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
            EC = np.load(filepath + self.file_string.format(**combination))
            EC = np.mean(np.sum(EC.f.arr_0, axis=0)[0])
            EC_list.append(EC/np.mean(EUL))

        legend = r'$\frac{\mathcal{K}_{EU}^{EB}}{\left\langle L_{EU}\right\rangle}$'
        ax.plot(alpha_list, EC_list, '.k', label=legend, ms=10)
        str2 = 'constrained' if c=='c' else 'unconstrained'
        str3 = 'synchronized' if f=='s' else 'localized'
        str4 = str3 + ' ' + str2 + r' flow $\beta={0}$'.format(beta)
        ax.set_title(str4, y=1.08, fontsize=15)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.legend(loc=2)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        plt.savefig('results/figures/ECvsAlpha.png')
        return

if __name__ == '__main__':
    B = BackupEurope('results/balancing/', 'data/')
#     B.plot_colormap()
#     B.plot_timeseries()
#     B.plot_timeseries_EU()
    B.plot_alpha()

