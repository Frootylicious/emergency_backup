import numpy as np
import matplotlib.pyplot as plt
import settings.settings as s
import settings.tools as t
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')


class FigurePlot():

    def __init__(self):
        # SET DATA
        self.EU = [np.load(s.iset_folder + s.iset_prefix + s.files[country]) for country in range(30)]
        self.nhours = len(self.EU[18]['L'])
        # LOADS
        self.L_EU = np.array([n['L'] for n in self.EU])
        self.avg_L_EU = np.mean(np.sum(self.L_EU, axis=0))
        self.avg_L_DE = np.mean(self.L_EU[s.country_dict['DE']])

    def set_data_no_solve(self, alpha, gamma):
        Gw_EU = np.empty((30, self.nhours))
        Gs_EU = np.empty((30, self.nhours))

        for n in range(30):
            L = self.L_EU[n]
            avg_L_n = np.mean(L)

            Gw_n_norm = self.EU[n]['Gw']
            Gs_n_norm = self.EU[n]['Gs']

            avg_Gw_n = alpha * gamma * avg_L_n
            avg_Gs_n = (1 - alpha) * gamma * avg_L_n

            Gw_n = avg_Gw_n * Gw_n_norm
            Gw_EU[n, :] = Gw_n

            Gs_n = avg_Gs_n * Gs_n_norm
            Gs_EU[n, :] = Gs_n

        # Mismatch for each node in each timestep
        D_n = np.array([Gw_EU[n] + Gs_EU[n] - self.L_EU[n] for n in range(30)])

        # Curtailment in each node
        C_n = np.copy(D_n)
        C_n[C_n < 0] = 0

        # Backup generation in each node in each timestep
        G_B_n = np.copy(D_n)
        G_B_n[G_B_n > 0] = 0
        G_B_n *= -1

        # Mismatch for whole EU
        # D_EU = np.sum([Gw_EU[n] + Gs_EU[n]- L_EU[n] for n in range(30)], axis=0)
        D_EU = np.sum(D_n, axis=0)

        # Curtailment for whole EU
        C_EU = np.copy(D_EU)
        C_EU[C_EU < 0] = 0

        # Backup generation for whole EU
        G_B_EU = np.copy(D_EU)
        G_B_EU[G_B_EU > 0] = 0
        G_B_EU *= -1

        self.G_B_EU = G_B_EU
        return(Gw_EU, Gs_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU)

    def Figure1(self, summer_week=30, winter_week=60):
        '''
        F41 consists of two backup timeseries of two weeks in the summer with zero and synchronized
        transmission.
        F41 is the same in the winter.
        '''

        N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=1.00))
        N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
        G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

        data_b0 = G_B_DE_b0 / self.avg_L_DE / 1000
        data_binf = G_B_DE_binf / self.avg_L_DE / 1000

        # Finding two weeks

        summer_week *= 7 * 24
        winter_week *= 7 * 24
        two_weeks = 24 * 7 * 2
        data_summer_b0 = data_b0[summer_week:summer_week + two_weeks]
        data_winter_b0 = data_b0[winter_week:winter_week + two_weeks]
        data_summer_binf = data_binf[summer_week:summer_week + two_weeks]
        data_winter_binf = data_binf[winter_week:winter_week + two_weeks]

        x = [0, len(data_summer_b0)]
        x_ticks = ['$t$', '$t$ + 2 weeks']

        fig1, (ax1) = plt.subplots(1, 1)
        ax1.plot(data_summer_b0, 'r', label=r'Summer $\beta^T=0$')
        ax1.plot(data_summer_binf, 'b', label=r'Summer $\beta^T=\infty$')

        fig2, (ax2) = plt.subplots(1, 1)

        ax2.plot(data_winter_b0, 'r', label=r'Winter $\beta^T=0$')
        ax2.plot(data_winter_binf, 'b', label=r'Winter $\beta^T=\infty$')

        ax1.axhline(0.7, color='grey', alpha=0.75, linestyle='--', label=r'$\approx 0.7$')
        ax2.axhline(0.7, color='grey', alpha=0.75, linestyle='--', label=r'$\approx 0.7$')

        ax1.set_xticks(x)
        ax2.set_xticks(x)
        ax1.set_xticklabels(x_ticks, ha="right", va="top", fontsize=20)
        ax2.set_xticklabels(x_ticks, ha="right", va="top", fontsize=20)
        ax1.set_xlim(x)
        ax2.set_xlim(x)
        ax1.set_ylim([0, 1.4])
        ax2.set_ylim([0, 1.4])
        ax1.set_ylabel(r'$\frac{G^B_n\left(t\right)}{\left<L_n\right>}$',
                    rotation=0, fontsize=20, labelpad=20)
        ax2.set_ylabel(r'$\frac{G^B_n\left(t\right)}{\left<L_n\right>}$',
                    rotation=0, fontsize=20, labelpad=20)
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')

        fig1.savefig(s.figures_folder + 'FIGURE4/' + 'F41.pdf')
        fig2.savefig(s.figures_folder + 'FIGURE4/' + 'F42.pdf')

        # plt.show()
        plt.close('all')

    def Figure2(self):
        '''
        F1 consists of two histograms of the distribution of the backup generation for zero and
        synchronized transmission.
        '''
        Gw_EU, Gs_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU = self.set_data_no_solve(0.8, 1.0)
        data_EU = G_B_EU / self.avg_L_EU

        country_id = s.country_dict['DE']
        data_DE = G_B_n[country_id] / np.mean(self.L_EU[country_id])
        xlabel = r'$G_n^B/\left<L_n \right>$'
        ylabel = r'$p(G_n^B)$'
        fig, (ax) = plt.subplots(1, 1, sharex=True)

        # Making histograms
        hist_EU, bins_EU = np.histogram(data_EU, density=True, bins=100)
        left_EU, right_EU = bins_EU[:-1], bins_EU[1:]
        X_EU = np.array([left_EU, right_EU]).T.flatten()
        Y_EU = np.array([hist_EU, hist_EU]).T.flatten()

        hist_DE, bins_DE = np.histogram(data_DE, density=True, bins=100)
        left_DE, right_DE = bins_DE[:-1], bins_DE[1:]
        X_DE = np.array([left_DE, right_DE]).T.flatten()
        Y_DE = np.array([hist_DE, hist_DE]).T.flatten()

        # Remove the 0-delta spike
        Y_EU = np.delete(Y_EU, [0, 1])
        X_EU = np.delete(X_EU, [0, 1])

        Y_DE = np.delete(Y_DE, [0, 1])
        X_DE = np.delete(X_DE, [0, 1])

        # Histogram plot
        # ax.plot(X, Y, '-k', alpha=0.6)
        ax.plot(X_EU, Y_EU, 'b', label=r'$\beta^T = \infty$')
        ax.plot(X_DE, Y_DE, 'r', label=r'$\beta^T = 0$')
        ax.set_ylabel(ylabel, fontsize=20, rotation=0, labelpad=25)
        ax.set_xlabel(xlabel, fontsize=20)
        # ax.set_xlim([0, 1.5])
        # ax.set_ylim([0, 1.3])
        ax.legend(fontsize=20)
        plt.savefig(s.figures_folder + 'FIGURE1/' + 'F1.pdf')
        plt.close('all')

    def Figure2_old(self, resolution=100):
        '''
        F21 is the backup generation as a function of alpha.
        F22 is the backup capacity with different quantiles as a function of alpha.
        '''
        # Data for subplot 1
        data_EU = np.empty(resolution)
        data_DE = np.empty(resolution)
        alpha_list = np.linspace(0, 1, resolution)

        for i, alpha in enumerate(alpha_list):
            Gw_EU, Gs_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU = self.set_data_no_solve(alpha, 1.0)
            country = 'DE'
            data_DE[i] = np.mean(G_B_n[s.country_dict[country]]) / self.avg_L_DE
            data_EU[i] = np.mean(G_B_EU) / self.avg_L_EU

        # Data for subplot 2
        n = 51
        alphas = np.linspace(0, 1.00, n)
        alpha = 1.00

        K_B_n_b0_q9 = np.empty(n)
        K_B_n_b0_q99 = np.empty(n)
        K_B_n_b0_q999 = np.empty(n)
        K_B_n_binf_q9 = np.empty(n)
        K_B_n_binf_q99 = np.empty(n)
        K_B_n_binf_q999 = np.empty(n)

        for i, alpha in tqdm(enumerate(alphas)):
            for beta in (0.00, np.inf):
                if beta == 0.00:
                    N = np.load(s.nodes_fullname.format(c='c', f='s', a=alpha, b=beta, g=1.00))
                    balancing_DE = N.f.balancing[s.country_dict['DE']]
                    K_B_n_b0_q9[i] = t.quantile(0.9, balancing_DE)
                    K_B_n_b0_q99[i] = t.quantile(0.99, balancing_DE)
                    K_B_n_b0_q999[i] = t.quantile(0.999, balancing_DE)
                else:
                    N = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=alpha, b=beta, g=1.00))
                    balancing_DE = N.f.balancing[s.country_dict['DE']]
                    K_B_n_binf_q9[i] = t.quantile(0.9, balancing_DE)
                    K_B_n_binf_q99[i] = t.quantile(0.99, balancing_DE)
                    K_B_n_binf_q999[i] = t.quantile(0.999, balancing_DE)

        K_B_n_b0_q9 /= self.avg_L_DE * 1000
        K_B_n_b0_q99 /= self.avg_L_DE * 1000
        K_B_n_b0_q999 /= self.avg_L_DE * 1000
        K_B_n_binf_q9 /= self.avg_L_DE * 1000
        K_B_n_binf_q99 /= self.avg_L_DE * 1000
        K_B_n_binf_q999 /= self.avg_L_DE * 1000

        # Plotting
        fig1, (ax1) = plt.subplots(1, 1)
        fig2, (ax2) = plt.subplots(1, 1)

        # Plot F21
        ax1.plot(alpha_list, data_DE, '-r', label=r'$\beta^T = 0$')
        ax1.plot(alpha_list, data_EU, '-b', label=r'$\beta^T = \infty$')
        ax1.set_ylabel(r'$\frac{\left< G^B_n \right>}{\left< L_n \right> }$',
                    fontsize=20, rotation=0, labelpad=20)
        ax1.set_xlabel(r'$\alpha$', fontsize=20)

        # ax1.set_xlim([0, 1])
        # ax1.set_ylim([0, 0.6])
        ax1.legend(loc='lower left')

        # Plot F22
        ax2.plot(alphas, K_B_n_b0_q9, ':r', label=r'$\beta^T=0, q=0.9$', alpha=0.75)
        ax2.plot(alphas, K_B_n_b0_q99, '-r', label=r'$\beta^T=0, q=0.99$', linewidth=2)
        ax2.plot(alphas, K_B_n_b0_q999, '--r', label=r'$\beta^T=0, q=0.999$', alpha=0.75)
        ax2.plot(alphas, K_B_n_binf_q9, ':b', label=r'$\beta^T=\infty, q=0.9$', alpha=0.75)
        ax2.plot(alphas, K_B_n_binf_q99, '-b', label=r'$\beta^T=\infty, q=0.99$', linewidth=2)
        ax2.plot(alphas, K_B_n_binf_q999, '--b', label=r'$\beta^T=\infty, q=0.999$', alpha=0.75)
        ax2.legend(loc='lower left', fontsize=13)
        ax2.set_ylabel(r'$\frac{K_n^B \left(q\right)}{\left<L_n\right>}$',
                    rotation=0, fontsize=20, labelpad=20)
        ax2.set_xlabel(r'$\alpha$', fontsize=20)
        # ax2.set_ylim([0, ])

        fig1.savefig(s.figures_folder + 'FIGURE2/' + 'F21.pdf')
        fig2.savefig(s.figures_folder + 'FIGURE2/' + 'F22.pdf')
        # plt.show()
        plt.close('all')

    def Figure3(self):
        '''
        F51 is the the average of the backup generation minus the backup capacity as a function of the
        backup capacity for both zero and synchronized transmission.
        F52 is the number of hours per year that the backup generation exceeds a given backup capacity.
        '''
        N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=1.00))
        N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
        G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

        G_B_DE_b0 /= (self.avg_L_DE * 1000)
        G_B_DE_binf /= (self.avg_L_DE * 1000)

        npoints = 151
        stop = 1.5

        avg_b0 = np.empty(npoints)
        avg_binf = np.empty(npoints)

        numbers_per_year_b0 = np.empty(npoints)
        numbers_per_year_binf = np.empty(npoints)

        x = np.linspace(0, stop, npoints)

        for i, beta_B_n in enumerate(x):
            K = beta_B_n
            data_b0 = [G_B_n - K if G_B_n - K >= 0 else 0 for G_B_n in G_B_DE_b0]
            data_binf = [G_B_n - K if G_B_n - K >= 0 else 0 for G_B_n in G_B_DE_binf]

            numbers_per_year_b0[i] = np.count_nonzero(data_b0) / 8
            numbers_per_year_binf[i] = np.count_nonzero(data_binf) / 8

            avg_b0[i] = np.mean(data_b0)
            avg_binf[i] = np.mean(data_binf)

        fig1, (ax1) = plt.subplots(1, 1)
        ax1.plot(x, avg_b0, 'r', label=r'$\beta^T=0$')
        ax1.plot(x, avg_binf, 'b', label=r'$\beta^T=\infty$')
        ax1.legend(loc='upper right')
        ax1.set_xlabel(r'$K^B_n/\left<L_n\right>$')
        ax1.set_ylabel(r'$\left< \max(G^B_n - K^B_n, 0) \right>$')
        ax1.set_xlim([0, stop])

        fig2, (ax2) = plt.subplots(1, 1)
        ax2.plot(x, numbers_per_year_b0, 'r', label=r'$\beta^T=0$')
        ax2.plot(x, numbers_per_year_binf, 'b', label=r'$\beta^T=\infty$')
        ax2.legend(loc='upper right')
        ax2.set_xlabel(r'$K^B_n/\left<L_n\right>$')
        ax2.set_ylabel(r'$T(G^B_n > K^B_n)/yr$')

        fig1.savefig(s.figures_folder + 'FIGURE5/' + 'F51.pdf')
        fig2.savefig(s.figures_folder + 'FIGURE5/' + 'F52.pdf')

        plt.close('all')

    def Figure4(self):
        '''
        F61 is the emergency backup energy as a function of the clustering time dt.
        F61_rnd is the same with randomized timeseries.
        F62 is the same as F61 with log-lin plot.
        F62_rnd ins the same with randomized timeseries.
        F63 is the same as F61 but divided by dt aswell, so we get the size per hour included.
        F63 is the same with randomized timeseries.
        F64 is the same as F63 in a log-log plot.
        F64_rnd is the same with randomized timeseries.
        F65 is the same as F61 in a log-log plot.
        F65_rnd is the same a with randomized timeseries.
        '''
        # Loading the nodes-objects from the solved networks.
        N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=1.00))
        N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        # Extracting the backup generation of DE.
        G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
        G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

        # Dividing by the average load to get relative units and converting to GW.
        G_B_DE_b0 /= (self.avg_L_DE * 1000)
        G_B_DE_binf /= (self.avg_L_DE * 1000)

        # Preparing the data-matrices.
        t_range = range(1, 2 * 7 * 24)
        K_range = np.linspace(0.25, 1, 4)

        # Whether or not to load data.
        load = True

        if load:
            load_data_b0 = np.load(s.figures_folder + 'FIGURE6/' + 'data_b0.npz')
            load_data_binf = np.load(s.figures_folder + 'FIGURE6/' + 'data_binf.npz')
            data_b0 = load_data_b0.f.arr_0
            data_binf = load_data_binf.f.arr_0
            load_data_b0_rnd = np.load(s.figures_folder + 'FIGURE6/' + 'data_b0_rnd.npz')
            load_data_binf_rnd = np.load(s.figures_folder + 'FIGURE6/' + 'data_binf_rnd.npz')
            data_b0_rnd = load_data_b0_rnd.f.arr_0
            data_binf_rnd = load_data_binf_rnd.f.arr_0
        else:
            data_b0 = np.empty((len(K_range), len(t_range)))
            data_binf = np.empty_like(data_b0)
            data_b0_rnd = np.empty((len(K_range), len(t_range)))
            data_binf_rnd = np.empty_like(data_b0)
            for i, K in tqdm(enumerate(K_range)):
                for j, dt in tqdm(enumerate(t_range)):

                    # Setting all negative to 0.
                    G_minus_K_b0 = [G - K if G - K >= 0 else 0 for G in G_B_DE_b0]
                    G_minus_K_binf = [G - K if G - K >= 0 else 0 for G in G_B_DE_binf]

                    # Randomizing
                    G_minus_K_b0_rnd = np.copy(G_minus_K_b0)
                    G_minus_K_binf_rnd = np.copy(G_minus_K_binf)
                    np.random.shuffle(G_minus_K_b0_rnd)
                    np.random.shuffle(G_minus_K_binf_rnd)

                    # Getting indexes for nonzero-elements
                    nonzero_b0 = np.nonzero(G_minus_K_b0)[0]
                    nonzero_binf = np.nonzero(G_minus_K_binf)[0]
                    nonzero_b0_rnd = np.nonzero(G_minus_K_b0_rnd)[0]
                    nonzero_binf_rnd = np.nonzero(G_minus_K_binf_rnd)[0]

                    # Number of qualified hours e.g. nonzero elements.
                    N_qh_b0 = len(nonzero_b0)
                    N_qh_binf = len(nonzero_binf)
                    N_qh_b0_rnd = len(nonzero_b0_rnd)
                    N_qh_binf_rnd = len(nonzero_binf_rnd)

                    # Summing all the sums.
                    data_b0[i, j] = np.sum([np.sum(G_minus_K_b0[t:t + dt])
                                            for t in nonzero_b0]) / N_qh_b0
                    data_binf[i, j] = np.sum([np.sum(G_minus_K_binf[t:t + dt])
                                            for t in nonzero_binf]) / N_qh_binf

                    data_b0_rnd[i, j] = np.sum([np.sum(G_minus_K_b0_rnd[t:t + dt])
                                            for t in nonzero_b0_rnd]) / N_qh_b0_rnd
                    data_binf_rnd[i, j] = np.sum([np.sum(G_minus_K_binf_rnd[t:t + dt])
                                            for t in nonzero_binf_rnd]) / N_qh_binf_rnd

            # Saving to files.
            np.savez_compressed(s.figures_folder + 'FIGURE6/' + 'data_b0.npz', data_b0)
            np.savez_compressed(s.figures_folder + 'FIGURE6/' + 'data_binf.npz', data_binf)
            np.savez_compressed(s.figures_folder + 'FIGURE6/' + 'data_b0_rnd.npz', data_b0_rnd)
            np.savez_compressed(s.figures_folder + 'FIGURE6/' + 'data_binf_rnd.npz', data_binf_rnd)

# PLOTTING NON-RANDOM ------------------------------------------------------------------------------
        # Plotting Figure6.1
        fig1, (ax1) = plt.subplots(1, 1)
        ax1.plot(t_range, data_b0[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax1.plot(t_range, data_b0[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax1.plot(t_range, data_b0[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax1.plot(t_range, data_b0[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax1.plot(t_range, data_binf[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax1.plot(t_range, data_binf[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax1.plot(t_range, data_binf[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax1.plot(t_range, data_binf[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax1.legend(loc='upper left', ncol=2)
        ax1.set_xlim([1, 2 * 7 * 24])
        ax1.set_xlabel(r'$\Delta t$')
        ax1.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig1.savefig(s.figures_folder + 'FIGURE6/' + 'F61.pdf')

        # Plotting Figure6.2
        fig2, (ax2) = plt.subplots(1, 1)
        ax2.plot(t_range, data_b0[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax2.plot(t_range, data_b0[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax2.plot(t_range, data_b0[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax2.plot(t_range, data_b0[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax2.plot(t_range, data_binf[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax2.plot(t_range, data_binf[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax2.plot(t_range, data_binf[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax2.plot(t_range, data_binf[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax2.set_yscale('log')

        ax2.legend(loc='lower center', ncol=2)
        ax2.set_xlim([1, 2 * 7 * 24])
        ax2.set_xlabel(r'$\Delta t$')
        ax2.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig2.savefig(s.figures_folder + 'FIGURE6/' + 'F62.pdf')

        # Plotting Figure 6.3
        fig3, (ax3) = plt.subplots(1, 1)
        ax3.plot(t_range, data_b0[0] / t_range, '--r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax3.plot(t_range, data_b0[1] / t_range, '--b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax3.plot(t_range, data_b0[2] / t_range, '--g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax3.plot(t_range, data_b0[3] / t_range, '--c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax3.plot(t_range, data_binf[0] / t_range, 'r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax3.plot(t_range, data_binf[1] / t_range, 'b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax3.plot(t_range, data_binf[2] / t_range, 'g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax3.plot(t_range, data_binf[3] / t_range, 'c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax3.legend(loc='upper center', ncol=2)
        ax3.set_xlim([1, 2 * 7 * 24])
        ax3.set_xlabel(r'$\Delta t$')
        ax3.set_ylabel(r'$E^{EB}_n(\Delta t)/\Delta t$')

        fig3.savefig(s.figures_folder + 'FIGURE6/' + 'F63.pdf')

        # Plotting Figure 6.4 (loglog of 6.3)
        fig4, (ax4) = plt.subplots(1, 1)
        ax4.plot(t_range, data_b0[0] / t_range, '--r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax4.plot(t_range, data_b0[1] / t_range, '--b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax4.plot(t_range, data_b0[2] / t_range, '--g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax4.plot(t_range, data_b0[3] / t_range, '--c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax4.plot(t_range, data_binf[0] / t_range, 'r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax4.plot(t_range, data_binf[1] / t_range, 'b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax4.plot(t_range, data_binf[2] / t_range, 'g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax4.plot(t_range, data_binf[3] / t_range, 'c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax4.set_yscale('log')
        ax4.set_xscale('log')

        ax4.legend(loc='lower left', ncol=2)
        ax4.set_xlim([1, 2 * 7 * 24])
        ax4.set_xlabel(r'$\Delta t$')
        ax4.set_ylabel(r'$E^{EB}_n(\Delta t)/\Delta t$')

        fig4.savefig(s.figures_folder + 'FIGURE6/' + 'F64.pdf')

        # Plotting Figure6.5 (loglog version of 6.2)
        fig5, (ax5) = plt.subplots(1, 1)
        ax5.plot(t_range, data_b0[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax5.plot(t_range, data_b0[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax5.plot(t_range, data_b0[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax5.plot(t_range, data_b0[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax5.plot(t_range, data_binf[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax5.plot(t_range, data_binf[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax5.plot(t_range, data_binf[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax5.plot(t_range, data_binf[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax5.set_yscale('log')
        ax5.set_xscale('log')

        ax5.legend(loc='upper left', ncol=2)
        ax5.set_xlim([1, 2 * 7 * 24])
        ax5.set_xlabel(r'$\Delta t$')
        ax5.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig5.savefig(s.figures_folder + 'FIGURE6/' + 'F65.pdf')

# PLOTTING RANDOM ------------------------------------------------------------------------------
        # Plotting Figure6.1
        fig1_rnd, (ax1_rnd) = plt.subplots(1, 1)
        ax1_rnd.plot(t_range, data_b0_rnd[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax1_rnd.plot(t_range, data_b0_rnd[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax1_rnd.plot(t_range, data_b0_rnd[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax1_rnd.plot(t_range, data_b0_rnd[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax1_rnd.plot(t_range, data_binf_rnd[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax1_rnd.plot(t_range, data_binf_rnd[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax1_rnd.plot(t_range, data_binf_rnd[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax1_rnd.plot(t_range, data_binf_rnd[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax1_rnd.legend(loc='upper left', ncol=2)
        ax1_rnd.set_xlim([1, 2 * 7 * 24])
        ax1_rnd.set_xlabel(r'$\Delta t$')
        ax1_rnd.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig1_rnd.savefig(s.figures_folder + 'FIGURE6/' + 'F61_rnd.pdf')

        # Plotting Figure6.2
        fig2_rnd, (ax2_rnd) = plt.subplots(1, 1)
        ax2_rnd.plot(t_range, data_b0_rnd[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax2_rnd.plot(t_range, data_b0_rnd[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax2_rnd.plot(t_range, data_b0_rnd[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax2_rnd.plot(t_range, data_b0_rnd[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax2_rnd.plot(t_range, data_binf_rnd[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax2_rnd.plot(t_range, data_binf_rnd[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax2_rnd.plot(t_range, data_binf_rnd[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax2_rnd.plot(t_range, data_binf_rnd[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax2_rnd.set_yscale('log')

        ax2_rnd.legend(loc='lower center', ncol=2)
        ax2_rnd.set_xlim([1, 2 * 7 * 24])
        ax2_rnd.set_xlabel(r'$\Delta t$')
        ax2_rnd.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig2_rnd.savefig(s.figures_folder + 'FIGURE6/' + 'F62_rnd.pdf')

        # Plotting Figure 6.3
        fig3_rnd, (ax3_rnd) = plt.subplots(1, 1)
        ax3_rnd.plot(t_range, data_b0_rnd[0] / t_range, '--r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax3_rnd.plot(t_range, data_b0_rnd[1] / t_range, '--b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax3_rnd.plot(t_range, data_b0_rnd[2] / t_range, '--g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax3_rnd.plot(t_range, data_b0_rnd[3] / t_range, '--c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax3_rnd.plot(t_range, data_binf_rnd[0] / t_range, 'r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax3_rnd.plot(t_range, data_binf_rnd[1] / t_range, 'b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax3_rnd.plot(t_range, data_binf_rnd[2] / t_range, 'g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax3_rnd.plot(t_range, data_binf_rnd[3] / t_range, 'c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax3_rnd.legend(loc='upper center', ncol=2)
        ax3_rnd.set_xlim([1, 2 * 7 * 24])
        ax3_rnd.set_xlabel(r'$\Delta t$')
        ax3_rnd.set_ylabel(r'$E^{EB}_n(\Delta t)/\Delta t$')

        fig3_rnd.savefig(s.figures_folder + 'FIGURE6/' + 'F63_rnd.pdf')

        # Plotting Figure 6.4 (loglog of 6.3)
        fig4_rnd, (ax4_rnd) = plt.subplots(1, 1)
        ax4_rnd.plot(t_range, data_b0_rnd[0] / t_range, '--r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax4_rnd.plot(t_range, data_b0_rnd[1] / t_range, '--b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax4_rnd.plot(t_range, data_b0_rnd[2] / t_range, '--g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax4_rnd.plot(t_range, data_b0_rnd[3] / t_range, '--c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax4_rnd.plot(t_range, data_binf_rnd[0] / t_range, 'r',
                label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax4_rnd.plot(t_range, data_binf_rnd[1] / t_range, 'b',
                label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax4_rnd.plot(t_range, data_binf_rnd[2] / t_range, 'g',
                label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax4_rnd.plot(t_range, data_binf_rnd[3] / t_range, 'c',
                label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax4_rnd.set_yscale('log')
        ax4_rnd.set_xscale('log')

        ax4_rnd.legend(loc='lower left', ncol=2)
        ax4_rnd.set_xlim([1, 2 * 7 * 24])
        ax4_rnd.set_xlabel(r'$\Delta t$')
        ax4_rnd.set_ylabel(r'$E^{EB}_n(\Delta t)/\Delta t$')

        fig4_rnd.savefig(s.figures_folder + 'FIGURE6/' + 'F64_rnd.pdf')

        # Plotting Figure6.5 (loglog version of 6.2)
        fig5, (ax5) = plt.subplots(1, 1)
        ax5.plot(t_range, data_b0_rnd[0], '--r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')
        ax5.plot(t_range, data_b0_rnd[1], '--b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=0$')
        ax5.plot(t_range, data_b0_rnd[2], '--g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=0$')
        ax5.plot(t_range, data_b0_rnd[3], '--c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=0$')

        ax5.plot(t_range, data_binf_rnd[0], 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')
        ax5.plot(t_range, data_binf_rnd[1], 'b', label=r'$K^B_n/\left<L_n\right> = 0.50, \beta^T=\infty$')
        ax5.plot(t_range, data_binf_rnd[2], 'g', label=r'$K^B_n/\left<L_n\right> = 0.75, \beta^T=\infty$')
        ax5.plot(t_range, data_binf_rnd[3], 'c', label=r'$K^B_n/\left<L_n\right> = 1.00, \beta^T=\infty$')

        ax5.set_yscale('log')
        ax5.set_xscale('log')

        ax5.legend(loc='upper left', ncol=2)
        ax5.set_xlim([1, 2 * 7 * 24])
        ax5.set_xlabel(r'$\Delta t$')
        ax5.set_ylabel(r'$E^{EB}_n(\Delta t)$')

        fig5.savefig(s.figures_folder + 'FIGURE6/' + 'F65_rnd.pdf')

    def Figure5(self):
        '''
        F51 is the emergency backup capacity as a function of alpha
        F72 is the emergency backup capacity as a function of the backup beta.
        F53 is a timeseries of backup generation and the emergency storage filling level.
        F53_rnd is the same as F53 but with randomized timeseries - the clustering is not present.
        F54 and F55 are the same as F53 but with different zoom.
        '''

        def F51(load=True):
            # Which alphas and gammas we want to look at.
            alpha_list = np.linspace(0, 1, 41)
            beta_b_list = np.linspace(0.25, 1, 4)

            if load:
                load_data_b0 = np.load(s.figures_folder + 'FIGURE7/' + 'data51_b0.npz')
                load_data_binf = np.load(s.figures_folder + 'FIGURE7/' + 'data51_binf.npz')
                data_b0 = load_data_b0.f.arr_0
                data_binf = load_data_binf.f.arr_0
            else:
                data_b0 = np.empty((len(beta_b_list), len(alpha_list)))
                data_binf = np.empty_like(data_b0)
                for i, alpha in tqdm(enumerate(alpha_list)):

                    # Loading the nodes-object.
                    N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=alpha, b=0.00, g=1.00))
                    N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=alpha, b=np.inf, g=1.00))

                    # Extracting the backup generation of DE.
                    G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
                    G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

                    # Dividing by the average load to get relative units and converting to GW.
                    G_B_DE_b0 /= (self.avg_L_DE * 1000)
                    G_B_DE_binf /= (self.avg_L_DE * 1000)

                    for j, beta_b in tqdm(enumerate(beta_b_list)):

                        data_b0[j, i] = t.storage_size_relative(G_B_DE_b0, beta_b)[0]
                        data_binf[j, i] = t.storage_size_relative(G_B_DE_binf, beta_b)[0]

                # Saving to files.
                np.savez_compressed(s.figures_folder + 'FIGURE5/' + 'data51_b0.npz', data_b0)
                np.savez_compressed(s.figures_folder + 'FIGURE5/' + 'data51_binf.npz', data_binf)

            # Plotting Figure7.1
            fig1, (ax1) = plt.subplots(1, 1)
            ax1.plot(alpha_list, data_b0[0], '--r', label=r'$\beta^B = 0.25, \beta^T=0$')
            ax1.plot(alpha_list, data_b0[1], '--b', label=r'$\beta^B  = 0.50, \beta^T=0$')
            ax1.plot(alpha_list, data_b0[2], '--g', label=r'$\beta^B  = 0.75, \beta^T=0$')
            ax1.plot(alpha_list, data_b0[3], '--c', label=r'$\beta^B  = 1.00, \beta^T=0$')

            ax1.plot(alpha_list, data_binf[0], 'r', label=r'$\beta^B  = 0.25, \beta^T=\infty$')
            ax1.plot(alpha_list, data_binf[1], 'b', label=r'$\beta^B  = 0.50, \beta^T=\infty$')
            ax1.plot(alpha_list, data_binf[2], 'g', label=r'$\beta^B  = 0.75, \beta^T=\infty$')
            ax1.plot(alpha_list, data_binf[3], 'c', label=r'$\beta^B  = 1.00, \beta^T=\infty$')

            ax1.legend(loc='upper right', ncol=2)
            ax1.set_yscale('log')
            ax1.set_xlabel(r'$\alpha$')
            ax1.set_ylabel(r'$\mathcal{K}^S_n$')
            # fig1.show()

            fig1.savefig(s.figures_folder + 'FIGURE7/' + 'F51.pdf')

        def F52(load=True):
            # Which alphas and gammas we want to look at.
            beta_b_list = np.linspace(0, 1, 11)

            if load:
                load_data_b0 = np.load(s.figures_folder + 'FIGURE5/' + 'data52_b0.npz')
                load_data_binf = np.load(s.figures_folder + 'FIGURE5/' + 'data52_binf.npz')
                data_b0 = load_data_b0.f.arr_0
                data_binf = load_data_binf.f.arr_0
            else:
                data_b0 = np.empty_like(beta_b_list)
                data_binf = np.empty_like(data_b0)

                # Loading the nodes-object.
                N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=1.00))
                N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

                # Extracting the backup generation of DE.
                G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
                G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

                # Dividing by the average load to get relative units and converting to GW.
                G_B_DE_b0 /= (self.avg_L_DE * 1000)
                G_B_DE_binf /= (self.avg_L_DE * 1000)

                for i, beta_b in tqdm(enumerate(beta_b_list)):

                    data_b0[i] = t.storage_size_relative(G_B_DE_b0, beta_b)[0]
                    data_binf[i] = t.storage_size_relative(G_B_DE_binf, beta_b)[0]

                # Saving to files.
                np.savez_compressed(s.figures_folder + 'FIGURE5/' + 'data52_b0.npz', data_b0)
                np.savez_compressed(s.figures_folder + 'FIGURE5/' + 'data52_binf.npz', data_binf)

            # Plotting Figure6.1
            fig1, (ax1) = plt.subplots(1, 1)
            ax1.plot(beta_b_list, data_b0, 'r', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=0$')

            ax1.plot(beta_b_list, data_binf, 'b', label=r'$K^B_n/\left<L_n\right> = 0.25, \beta^T=\infty$')

            ax1.legend(loc='upper right', ncol=2)
            ax1.set_yscale('log')
            ax1.set_xlabel(r'$\beta^B$')
            ax1.set_ylabel(r'$\mathcal{K}^S_n$')

            fig1.savefig(s.figures_folder + 'FIGURE5/' + 'F52.pdf')

        def F53_F54_F55(beta_b=0.55):
            K = beta_b

            N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=1.00))
            N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

            # Extracting the backup generation of DE.
            G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
            G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

            # Dividing by the average load to get relative units and converting to GW.
            G_B_DE_b0 /= (self.avg_L_DE * 1000)
            G_B_DE_binf /= (self.avg_L_DE * 1000)
            G_B_DE_binf_rnd = np.copy(G_B_DE_binf)
            np.random.shuffle(G_B_DE_binf_rnd)

            G_minus_K_b0 = [G - K if G - K >= 0 else 0 for G in G_B_DE_b0]
            G_minus_K_binf = [G - K if G - K >= 0 else 0 for G in G_B_DE_binf]
            G_minus_K_binf_rnd = [G - K if G - K >= 0 else 0 for G in G_B_DE_binf_rnd]


            a_b0 = t.storage_size_relative(G_B_DE_b0[:], K)
            a_binf = t.storage_size_relative(G_B_DE_binf[:], K)
            a_binf_rnd = t.storage_size_relative(G_B_DE_binf_rnd[:], K)

            fig3, (ax3) = plt.subplots(1, 1)
            ax3.plot(G_minus_K_binf, label=r'$G^B_n(t) - K^B_n$')
            ax3.plot(a_binf[1], label=r'$S_n(t)$')
            ax3.legend(loc='lower left')
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$E/\left<L_n\right>$')
            ax3.set_xlim((0, len(G_minus_K_binf)))
            fig3.savefig(s.figures_folder + 'FIGURE7/' + 'F53.pdf')

            fig3_rnd, (ax3_rnd) = plt.subplots(1, 1)
            ax3_rnd.plot(G_minus_K_binf_rnd, label=r'$G^B_n(t) - K^B_n$')
            ax3_rnd.plot(a_binf_rnd[1], label=r'$S_n(t)$')
            ax3_rnd.legend(loc='lower left')
            ax3_rnd.set_xlabel(r'$t$')
            ax3_rnd.set_ylabel(r'$E/\left<L_n\right>$')
            ax3_rnd.set_xlim((0, len(G_minus_K_binf)))
            fig3_rnd.savefig(s.figures_folder + 'FIGURE7/' + 'F53_rnd.pdf')

            time4 = [53500 - 2 * 7 * 24, 53500 + 2 * 7 * 24]

            fig4, (ax4) = plt.subplots(1, 1)
            ax4.plot(G_minus_K_binf[time4[0]:time4[1]], label=r'$G^B_n(t) - K^B_n$')
            ax4.plot(a_binf[1][time4[0]:time4[1]], label=r'$S_n(t)$')
            ax4.legend(loc='lower left')
            ax4.set_xlabel(r'$t$')
            ax4.set_ylabel(r'$E/\left<L_n\right>$')
            ax4.set_xlim((0, time4[1] - time4[0]))
            fig4.savefig(s.figures_folder + 'FIGURE7/' + 'F54.pdf')

            time5 = [53500 - 134, 53500 - 62]

            fig5, (ax5) = plt.subplots(1, 1)
            ax5.plot(G_minus_K_binf[time5[0]:time5[1]], label=r'$G^B_n(t) - K^B_n$')
            ax5.plot(a_binf[1][time5[0]:time5[1]], label=r'$S_n(t)$')
            ax5.legend(loc='lower left')
            ax5.set_xlabel(r'$t$')
            ax5.set_ylabel(r'$E/\left<L_n\right>$')
            ax5.set_xlim((0, time5[1] - time5[0]))
            fig5.savefig(s.figures_folder + 'FIGURE7/' + 'F55.pdf')
            plt.close('all')

        F51()
        F52()
        F53_F54_F55(beta_b=0.55)

    def Figure8(self):
        '''
        F81 is the emergency storage capacity as a function of gamma.
        '''

        def F81(load=False):
            # Which alphas and gammas we want to look at.
            alpha = 0.80
            gamma_list = np.linspace(0, 2, 51)
            beta_b_list = np.linspace(0.25, 1, 4)

            if load:
                load_data_b0 = np.load(s.figures_folder + 'FIGURE8/' + 'data81_b0.npz')
                load_data_binf = np.load(s.figures_folder + 'FIGURE8/' + 'data81_binf.npz')
                data_b0 = load_data_b0.f.arr_0
                data_binf = load_data_binf.f.arr_0
            else:
                data_b0 = np.empty((len(beta_b_list), len(gamma_list)))
                data_binf = np.empty_like(data_b0)
                for i, gamma in tqdm(enumerate(gamma_list)):

                    # Loading the nodes-object.
                    N_b0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0.00, g=gamma))
                    N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=gamma))

                    # Extracting the backup generation of DE.
                    G_B_DE_b0 = N_b0.f.balancing[s.country_dict['DE']]
                    G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

                    # Dividing by the average load to get relative units and converting to GW.
                    G_B_DE_b0 /= (self.avg_L_DE * 1000)
                    G_B_DE_binf /= (self.avg_L_DE * 1000)

                    for j, beta_b in tqdm(enumerate(beta_b_list)):

                        data_b0[j, i] = t.storage_size_relative(G_B_DE_b0, beta_b)[0]
                        data_binf[j, i] = t.storage_size_relative(G_B_DE_binf, beta_b)[0]

                # Saving to files.
                np.savez_compressed(s.figures_folder + 'FIGURE8/' + 'data81_b0.npz', data_b0)
                np.savez_compressed(s.figures_folder + 'FIGURE8/' + 'data81_binf.npz', data_binf)

            # Plotting Figure6.1
            fig1, (ax1) = plt.subplots(1, 1)
            ax1.plot(gamma_list, data_b0[0], '--r', label=r'$\beta^B = 0.25, \beta^T=0$')
            ax1.plot(gamma_list, data_b0[1], '--b', label=r'$\beta^B  = 0.50, \beta^T=0$')
            ax1.plot(gamma_list, data_b0[2], '--g', label=r'$\beta^B  = 0.75, \beta^T=0$')
            ax1.plot(gamma_list, data_b0[3], '--c', label=r'$\beta^B  = 1.00, \beta^T=0$')

            ax1.plot(gamma_list, data_binf[0], 'r', label=r'$\beta^B  = 0.25, \beta^T=\infty$')
            ax1.plot(gamma_list, data_binf[1], 'b', label=r'$\beta^B  = 0.50, \beta^T=\infty$')
            ax1.plot(gamma_list, data_binf[2], 'g', label=r'$\beta^B  = 0.75, \beta^T=\infty$')
            ax1.plot(gamma_list, data_binf[3], 'c', label=r'$\beta^B  = 1.00, \beta^T=\infty$')

            ax1.legend(loc='upper right', ncol=2)
            ax1.set_yscale('log')
            ax1.set_xlabel(r'$\gamma$')
            ax1.set_ylabel(r'$\mathcal{K}^S_n$')
            ax1.set_ylim([ax1.get_ylim()[0], 1e7])

            fig1.savefig(s.figures_folder + 'FIGURE8/' + 'F81.pdf')
            plt.close('all')
        F81(load=True)

    def Figure9(self):
        '''
        F91 is the sum of the transmission capacity time the link length in the unit of GW/1000 km and as
        a function of alpha.
        F92 is the same as F91 but without the multiplication of the link length.
        '''

        alpha_list = np.linspace(0, 1, 51)
        distances = np.array(s.link_distances)

        # alpha = 0.80

#         copper_name = s.results_folder + 'copperflows/' + 'copperflow_a{a:.2f}_g1.00.npy'

        result1 = np.empty_like(alpha_list)
        result2 = np.empty_like(alpha_list)

        for i, alpha in tqdm(enumerate(alpha_list)):

            # F_copper = np.load(copper_name.format(a=alpha))
            F_copper = np.load(s.links_fullname.format(c='c', f='s', a=alpha, g=1.00, b=np.inf))
            F_copper = F_copper.f.arr_0

            F_abs = np.abs(F_copper)

            K_T_l = np.array([t.quantile_old(0.99, link) for link in F_abs])

            result1[i] = np.sum(K_T_l * distances)
            result2[i] = np.sum(K_T_l)


        result1 /= 1000000
        result2 /= 1000

        fig1, (ax1) = plt.subplots(1, 1)
        ax1.plot(alpha_list, result1, label=r'$\sum_l \kappa^T_l dl$')

        # ax1.legend(loc='upper center')
        ax1.set_xlabel(r'$\alpha$', fontsize=20)
        ax1.set_ylabel(r'$\sum_l \kappa^T_l dl$', fontsize=15)
        ax1.set_xlim((0, 1))
        fig1.savefig(s.figures_folder + 'FIGURE9/' + 'F91.pdf')
        plt.close('all')

        fig2, (ax2) = plt.subplots(1, 1)
        ax2.plot(alpha_list, result2, label=r'$\sum_l \kappa^T_l$')

        # ax2.legend(loc='upper center')
        ax2.set_xlabel(r'$\alpha$', fontsize=20)
        ax2.set_ylabel(r'$\sum_l \kappa^T_l$', fontsize=15)
        ax2.set_xlim((0, 1))
        fig2.savefig(s.figures_folder + 'FIGURE9/' + 'F92.pdf')
        plt.close('all')

    def Figure10(self):
        '''
        Three figures:
        F101 is the mean of the backup generation as a function og the transmission beta
        F102 is the 90 %, 99 % and 99.9 % quantile of the backup generation as a function of the
        transmission beta.
        F103 is the emergency backup storage size, with a backup beta of 0.7 as a function of the
        transmission beta.
        '''

        alpha = 0.80
        gamma = 1.00
        beta_T_list = np.linspace(0, 1.5, 16)

        data1 = np.empty_like(beta_T_list)
        data2 = np.empty((len(beta_T_list), len(beta_T_list)))
        data3 = np.empty_like(beta_T_list)

        for i, beta_T in tqdm(enumerate(beta_T_list)):
#             N = t.load_remote_network('c_s_a0.80_g1.00_b{:.2f}'.format(beta_T), N_or_F='N')
            N = np.load(s.nodes_fullname.format(c='c', f='s', a=alpha, b=beta_T, g=gamma))

            # Extracting the backup generation of DE.
            G_B_DE = N.f.balancing[s.country_dict['DE']]

            # Dividing by the average load to get relative units and converting to GW.
            G_B_DE /= (self.avg_L_DE * 1000)

            data1[i] = np.mean(G_B_DE)
            data2[0, i] = t.quantile_old(0.9, G_B_DE)
            data2[1, i] = t.quantile_old(0.99, G_B_DE)
            data2[2, i] = t.quantile_old(0.999, G_B_DE)
            data3[i] = t.storage_size_relative(G_B_DE, 0.7)[0]

        fig1, (ax1) = plt.subplots(1, 1)
        ax1.plot(beta_T_list, data1)
        ax1.set_xlabel(r'$\beta^T$', fontsize=20)
        ax1.set_ylabel(r'$\langle G^B_n \rangle / \langle L_n \rangle$', fontsize=20)
        ax1.set_xlim([beta_T_list[0], beta_T_list[-1]])
        fig1.savefig(s.figures_folder + 'FIGURE10/' + 'F101.pdf')

        fig2, (ax2) = plt.subplots(1, 1)
        ax2.plot(beta_T_list, data2[0], label=r'$\kappa^B_n = 0.9$')
        ax2.plot(beta_T_list, data2[1], label=r'$\kappa^B_n = 0.99$')
        ax2.plot(beta_T_list, data2[2], label=r'$\kappa^B_n = 0.999$')
        ax2.set_xlabel(r'$\beta^T$', fontsize=20)
        ax2.set_ylabel(r'$\kappa^B_n / \langle L_n \rangle$', fontsize=20)
        ax2.legend(loc='upper right')
        ax2.set_xlim([beta_T_list[0], beta_T_list[-1]])
        fig2.savefig(s.figures_folder + 'FIGURE10/' + 'F102.pdf')

        fig3, (ax3) = plt.subplots(1, 1)
        ax3.plot(beta_T_list, data3)
        ax3.plot(beta_T_list, data3, '.')
        ax3.set_xlabel(r'$\beta^T$', fontsize=20)
        ax3.set_ylabel(r'$\kappa^S_n$', fontsize=20)
        ax3.set_xlim([beta_T_list[0], beta_T_list[-1]])
        fig3.savefig(s.figures_folder + 'FIGURE10/' + 'F103.pdf')
#     plt.show()
        plt.close('all')

    def F5_inset(self):

        K = 0.75

        N_binf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        # Extracting the backup generation of DE.
        G_B_DE_binf = N_binf.f.balancing[s.country_dict['DE']]

        # Dividing by the average load to get relative units and converting to GW.
        G_B_DE_binf /= (self.avg_L_DE * 1000)

        G_minus_K_binf = [G - K if G - K >= 0 else 0 for G in G_B_DE_binf]

        a_binf = t.storage_size_relative(G_B_DE_binf[:], K)

        x_ticks = np.linspace(0, 8, self.nhours)

        fig3, (ax3) = plt.subplots(1, 1)
        ax3.plot(x_ticks, G_minus_K_binf, label=r'$G^B_n(t) - K^B_n$')
        ax3.plot(x_ticks, a_binf[1], label=r'$S_n(t)$')
        ax3.legend(loc='lower left')
        ax3.set_xlabel(r'$t$')
        ax3.set_ylabel(r'$E/\left<L_n\right>$')
        ax3.set_xlim((0, 8))
#         fig3.savefig(s.figures_folder + 'FIGURE7/' + 'F53.pdf')

        time5 = [53500 - 134, 53500 - 62]

#         axins = zoomed_inset_axes(ax3, 3, loc=3) # zoom-factor: 2.5, location: upper-left
        axins = inset_axes(ax3, 3,2 , loc=3, bbox_to_anchor=(0.12, 0.30), bbox_transform=ax3.figure.transFigure) # no zoom
        axins.set_xlim(x_ticks[time5[0]], x_ticks[time5[1]]) # apply the x-limits
        axins.set_ylim(-3.5, 0.5) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax3, axins, loc1=4, loc2=1, fc="none", ec="0.5")
        axins.plot(x_ticks, G_minus_K_binf)
        axins.plot(x_ticks, a_binf[1])
        plt.show()
