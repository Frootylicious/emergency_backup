#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
# matplotlib.use('Agg')
import sys
import os, os.path
import settings as s
import numpy as np
import colormaps as cmaps
import matplotlib.pyplot as plt
import regions.classes as cl
import tools as to
from itertools import product
from data_solving import Data


'''
TODO:
'''


class BackupEurope(object):
    """ Class to calculate the desired emergency backup capacities.

    Paths to the different folders can be set in 'settings.py'.
    """
    def __init__(self):
        """ Initially calls '_read_from_file()' which sets all present solved networks in the
        results-folder to the list 'all_combinations'.
        """
        # Saving all combinations present from files.
        self.all_combinations = to.read_from_file()
        self.chosen_combinations = to.get_chosen_combinations()

        # Creating results- and figures folder.
        if not os.path.exists(s.results_folder):
            os.mkdir(s.results_folder)
        if not os.path.exists(s.figures_folder):
            os.mkdir(s.figures_folder)

        # Loading the loads from the ISET-data.
        self.loads = np.array([np.load('%sISET_country_%s.npz'\
                % (s.iset_folder, s.countries[node]))['L']\
                for node in range(len(s.countries))])

    def get_chosen_combinations(self, **kwargs):
        self.chosen_combinations = to.get_chosen_combinations(**kwargs)
    

# CALCULATING EMERGENCY BACKUP ---------------------------------------------------------------------

    def _calculate_all_EC(self, save_path=s.EBC_folder, quantile=0.99):
        # Calculating all EC in self.all_combinations.
        for combination in self.all_combinations:
            self._calculate_EC(combination)
        return

    def _calculate_chosen_EC(self, save_path=s.EBC_folder,  quantile=0.99):
        # Calculating EBC in self.chosen_combinations.
        for combination in self.chosen_combinations:
            self._calculate_EC(combination)


    def _calculate_EC(self,
                      combination_dict,
                      save_path=s.EBC_folder,
                      quantile=0.99):
        '''
        Function that calculates emergency storage capacities for all countries
        in the file given by the combination-dictionary and saves them to files.

        The saved file is a 30 numpy array with the emergency backup capacity

        Saves in the form:
        EC_{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}.npy

        Args:
            combination_dict [dictionary]: a dictionary containing the desired network parameters.
            save_paths [string]: the folder where the EBC should be saved to.
            quantile [float]: the value of the quantile.

        '''
        print combination_dict
        # Check if path exists. Create it if not.
        if not os.path.exists(s.EBC_folder):
            os.mkdir(s.EBC_folder)
        if os.path.isfile(s.EBC_fullname.format(**combination_dict)):
            print('EC-file {0} already exists - skipping.'.format(combination_dict))
        else:
#             combination_caps = np.zeros(len(s.countries))
            nodes = np.load(s.nodes_fullname.format(**combination_dict))
            balancing = nodes['balancing']
#             for i, country_backup in enumerate(balancing):
#                 combination_caps[i] = to.storage_size(country_backup, quantile)[0]
            storage_size = to.storage_size(np.sum(balancing, axis=0), q=0.99)[0]
            np.savez(s.EBC_fullname.format(**combination_dict), storage_size)
#             np.savez(s.EBC_fullname.format(**combination_dict), combination_caps)
            print('Saved EC-file: {0}'.format(combination_dict))
        return

# PLOTTING -----------------------------------------------------------------------------------------

    def plot_colormap(self, f='s', c='c', b=1.00, a_amount=11, g_amount=11):
        """
        Plot a colormap with emergency backup capacity as a function of alpha and gamma.
        
        First the desired combinations are chosen and the emergency backup capacities are
        calculated. 
        Then a matrix of NaNs is initiated with the given alpha and gamma spacing.
        The matrix is filled with EBC-values available and is still NaN where the EBC-values are
        missing.

        This results in a colormap where there are white spaces when the EBC-value is missing. This
        way a plot can be generated with incomplete data.

        Args:
            f [string]:
            c [string]:
            b [float]: the fixed beta value for this plot.
            a_amount [integer]: number of values on the alpha-axis.
            g_amount [integet]: number of values on the gamma_axis.
        """
        # Preparing the lists for alpha and gamma values.
        alpha_list = np.linspace(0, 1, a_amount)
        gamma_list = np.linspace(0, 2, g_amount)
        # Only get the combinations with the wanted f and c.
        self.get_chosen_combinations(f=f, c=c, b=b, a=alpha_list, g=gamma_list)
        # Calculate the emergency capacities for the wanted values.
        self._calculate_chosen_EC()
        # Finding the space between first and second value in the lists.
        da = float('{0:.2f}'.format(np.diff(alpha_list)[0]))
        dg = float('{0:.2f}'.format(np.diff(gamma_list)[0]))
        EC_matrix = np.empty((len(alpha_list), len(gamma_list)))
        EC_matrix[:] = np.nan
        # Calculating the mean of the sum of the loads for europe.
        mean_sum_loads = np.mean(np.sum(self.loads, axis=0)) * 1000
        number_of_nans = 0
        # This loop loads the EBC from the saved files and puts them into the EBC-matrix.
        for i, a in enumerate(alpha_list):
            for j, g in enumerate(gamma_list):
                load_dict = {'f':f, 'c':c, 'b':b, 'a':a, 'g':g}
                if os.path.isfile(s.EBC_fullname.format(**load_dict)):
                    # If the file is present and therefore calculated.
                    EC = np.load(s.EBC_fullname.format(**load_dict))['arr_0']
                    EC_matrix[i, j] = np.sum(EC) / mean_sum_loads
                else:
                    # If file is not present.
                    number_of_nans += 1

        if number_of_nans > 0:
            print("{0} files didn't exist - set as NaN".format(number_of_nans))

        # Important to do this to be able to plot the NaNs
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
        save_str = 'cmap_f{0}_c{1}_b{2:.2f}.png'.format(f, c, b)
        plt.savefig(s.figures_folder + save_str)
        print('Saved file "{0}".'.format(save_str))
        plt.close()
        return

    def plot_timeseries_EU(self, a=0.80, g=1.00, b=1.00, c='c', f='s'):
        """
        Plot a timeseries of a combined EU with load, solar generation, wind generation, 99 %
        quantile of balancing, export, import and curtailment.

        Args:
            a [float]: value of alpha.
            g [float]: value of gamma.
            b [float]: value of beta.
        """
        N = cl.Nodes(load_filename=s.nodes_fullname.format(c=c, f=f, a=a, g=g, b=b),
                     files=s.files,
                     load_path='',
                     path=s.iset_folder,
                     prefix=s.iset_prefix)

        timeseries = np.zeros((30, N[0].nhours))
        q1 = np.zeros((30, N[0].nhours))

        for i, n in enumerate(N):
            timeseries[i] = n.load - n.get_solar() - n.get_wind() - to.quantile(0.99,
                    n.get_balancing()) + n.get_export() - n.get_import() + n.get_curtailment()

        EUL_avg = np.mean(np.sum([x.load for x in N], axis=0))

        K_EB = np.load(s.EBC_fullname.format(c='c', f='s', a=a, g=g, b=b))
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
#         ax.plot(q2, 'r')
        ax.set_title(title_str1 + title_str2, fontsize=20, y=0.9, x=0.3)
        fig.text(x=0.1, y=0.7, s=txt_str1.format(K_EB), fontsize=20)
        fig.text(x=0.1, y=0.6, s=txt_str2, fontsize=15)
        plt.tight_layout()
        plt.show()
        plt.savefig('results/figures/tsEU_{0}.png'.format(s.nodes_name.format(c=c, f=f, a=a, g=g, b=b)))
        plt.close()
        return


    def plot_storage_EU(self, a=0.80, g=1.00, b=1.00, c='c', f='s'):
        N = cl.Nodes(load_filename=s.nodes_fullname.format(c=c, f=f, a=a, g=g, b=b),
                     files=s.files,
                     load_path='',
                     path=s.iset_folder,
                     prefix=s.iset_prefix)

        storage_timeseries = np.zeros((30, N[0].nhours))
        backup_offset = np.zeros((30, N[0].nhours))
        balancing_timeseries_EU = np.zeros(N[0].nhours)


        for i, n in enumerate(N):
            balancing_timeseries_EU += n.get_balancing()
            (storage_size, storage_timeseries[i], backup_offset[i]) = to.storage_size(n.get_balancing(), q=0.99)

        (storage_EU, storage_timeseries2, backup_offset2) = to.storage_size(balancing_timeseries_EU)

        EUL_avg = np.mean(np.sum([x.load for x in N], axis=0))
        storage_timeseries = np.sum(storage_timeseries, axis=0) / EUL_avg
        backup_offset = np.sum(backup_offset, axis=0) / EUL_avg
        storage_timeseries2 /= EUL_avg
        backup_offset2 /= EUL_avg
        fig, (ax1, ax2, ax3)  = plt.subplots(3, 1)
        ax1.plot(storage_timeseries, 'r')
        ax1.plot(backup_offset)
        ax1.set_title('Emergency backup capacity for constrained synchronized network:\n' +
                r'$\alpha=0.8, \gamma=1, \beta=1$')
        ax1.set_xlim([0, len(storage_timeseries)])
        ax2.plot(storage_timeseries2, 'r', label='Emergency backup need')
        ax2.plot(backup_offset2, label='$B_{EU}-B_{EU}(q=99\%)$')
        ax2.set_xlim([0, len(storage_timeseries)])
        ax2.legend(loc=2)
        ax2.set_ylabel(r'$K_{EU}^{EB}/\left<L_{EU}\right>$')
        ax2.set_xlabel('Hours')
        ax3.plot(storage_timeseries2[27275:27400], 'r')
        ax3.plot(backup_offset2[27275:27400])
        ax3.plot([0, 27400-27275-1], [0, 0], 'k--', lw=1)
        ax3.set_xlim([0, 27400-27275-1])
        ax3.set_xlabel('Hours $-27400$')
        plt.savefig('results/figures/storageEU.pdf')


    def plot_storage_DK(self, a=0.80, g=1.00, b=1.00, c='c', f='s'):
        N = cl.Nodes(load_filename=s.nodes_fullname.format(c=c, f=f, a=a, g=g, b=b),
                        files=s.files,
                        load_path='',
                        path=s.iset_folder,
                        prefix=s.iset_prefix)

        storage_timeseries = np.zeros(N[0].nhours)
        backup_offset = np.zeros(N[0].nhours)

        (storage_size, storage_timeseries, backup_offset) = to.storage_size(N[21].get_balancing(), q=0.99)

        fig, (ax)  = plt.subplots(1, 1, sharex=True)
        ax.plot(storage_timeseries, 'r')
        ax.plot(backup_offset)
        plt.show()



    def plot_alpha(self, g=1.00, b=1.00, c='c', f='s'):
        """Plot the EBC as a function of alpha with gamma and beta fixed
        Args:
            gamma [float]: value of the fixed gamma.
            beta [float]: value of the fixed beta.
            c [string]: Either 'c' or 'u' for constrained or unconstrained flowscheme.
            f [string]: Either 's' or 'l' for Synchronized or Localized flowscheme.
        """

        self.get_chosen_combinations(g=g, b=b, c=c, f=f)
        self._calculate_chosen_EC()
        a_list = []
        EC_list = []
        EUL = np.array([np.load('%sISET_country_%s.npz'\
                % (s.iset_folder, s.countries[node]))['L']\
                for node in range(len(s.countries))])
        EUL = np.sum(EUL, axis=0) * 1000
        for combination in self.chosen_combinations:
            a_list.append(combination['a'])
            EC = np.load(s.EBC_fullname.format(**combination))
            EC = np.mean(np.sum(EC.f.arr_0, axis=0))
            EC_list.append(EC/np.mean(EUL))

        legend = r'$\frac{\mathcal{K}_{EU}^{EB}}{\left\langle L_{EU}\right\rangle}$'
        fig, (ax)  = plt.subplots(1, 1, sharex=True)
        ax.plot(a_list, EC_list, '.k', label=legend, ms=10)
        str2 = 'constrained' if c=='c' else 'unconstrained'
        str3 = 'synchronized' if f=='s' else 'localized'
        str4 = str3 + ' ' + str2 + r' flow $\beta={0}$'.format(b)
        ax.set_title(str4, y=1.08, fontsize=15)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.legend(loc=2)
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        str5 = s.nodes_name[:7] + s.nodes_name[16:]
        plt.savefig('results/figures/alpha_{0}.png'.format(str5.format(g=g, b=b, c=c, f=f)))
        plt.close()
        return


if __name__ == '__main__':
    B = BackupEurope()
    B.plot_storage_EU()
#     B.plot_colormap()
#     B.plot_timeseries()
#     B.plot_timeseries_EU()
#     B.plot_alpha()


