import numpy as np
import matplotlib.pyplot as plt
import settings as s
import tools as t
# import seaborn as sns

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')

load_iset = True

# SET DATA ---------------------------------------------------------------------------------------
if load_iset:
    EU = [np.load(s.iset_folder + s.iset_prefix + s.files[country]) for country in range(30)]
    alpha = 0.8
    gamma = 1
    nhours = len(EU[18]['L'])

    Gw_EU = np.empty((30, nhours))
    Gs_EU = np.empty((30, nhours))
    L_EU = np.empty((30, nhours))


def set_data(alpha, gamma):
    for n in range(30):

        L = EU[n]['L']
        L_EU[n, :] = L
        avg_L_n = np.mean(L)

        Gw_n_norm = EU[n]['Gw']
        Gs_n_norm = EU[n]['Gs']

        avg_Gw_n = alpha * gamma * avg_L_n
        avg_Gs_n = (1 - alpha) * gamma * avg_L_n

        Gw_n = avg_Gw_n * Gw_n_norm
        Gw_EU[n, :] = Gw_n

        Gs_n = avg_Gs_n * Gs_n_norm
        Gs_EU[n, :] = Gs_n

    avg_L_EU = np.mean(np.sum(L_EU, axis=0))

    # Mismatch for each node in each timestep
    D_n = np.array([Gw_EU[n] + Gs_EU[n] - L_EU[n] for n in range(30)])

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
    return(Gw_EU, Gs_EU, L_EU, avg_L_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU)


# FIGURE 1 ---------------------------------------------------------------------------------------

def G_B(name):
    if name == 'EU':
        data = G_B_EU / avg_L_EU
        xlabel = r'$G_{{EU}}^B/\left<L_{{EU}} \right>$'
        ylabel = r'$p(G_{{EU}}^B)$'
    else:
        country_id = s.country_dict[name]
        data = G_B_n[country_id] / np.mean(L_EU[country_id])
        xlabel = r'$G_n^B/\left<L_n \right>$'
        ylabel = r'$P(G_n^B)$'
    fig, (ax) = plt.subplots(1, 1, sharex=True)
    hist, bins = np.histogram(data, density=True, bins=100)
    left, right = bins[:-1], bins[1:]
    X = np.array([left, right]).T.flatten()
    Y = np.array([hist, hist]).T.flatten()

    # Remove the 0-delta spike
    Y = np.delete(Y, [0, 1])
    X = np.delete(X, [0, 1])

    # Histogram plot
    # ax.plot(X, Y, '-k', alpha=0.6)
    ax.plot(X, Y)
    ax.set_title(r'$G_B$ - {} - $\gamma_n=\gamma=1, \alpha_n=\alpha=0.8$'.format(name), fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_xlim([0, 2.0])
    ax.set_ylim([0, 1.5])
    plt.savefig(s.figures_folder + 'FIGURE1/' + 'G_B_{}.png'.format(name))
    plt.close()


def Figure1():
    Gw_EU, Gs_EU, L_EU, avg_L_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU = set_data(0.8, 1.0)
    data_EU = G_B_EU / avg_L_EU

    country_id = s.country_dict['DE']
    data_DE = G_B_n[country_id] / np.mean(L_EU[country_id])
    xlabel = r'$G_n^B/\left<L_n \right>$'
    ylabel = r'$p(G_n^B)$'
    fig, (ax) = plt.subplots(1, 1, sharex=True)

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
    plt.close()


def D(name):
    if name == 'EU':
        data = D_EU / avg_L_EU
        xlabel = r'$\Delta_{{EU}}^B/\left<L_{{EU}} \right>$'
        ylabel = r'$P(\Delta_{{EU}}^B)$'
    else:
        country_id = s.country_dict[name]
        data = D_n[country_id] / np.mean(L_EU[country_id])
        xlabel = r'$\Delta_n^B/\left<L_n \right>$'
        ylabel = r'$P(\Delta_n^B)$'
    fig, (ax) = plt.subplots(1, 1)
    hist, bins = np.histogram(data, density=True, bins=100)
    left, right = bins[:-1], bins[1:]
    X = np.array([left, right]).T.flatten()
    Y = np.array([hist, hist]).T.flatten()
    # print(data)

    # Remove the 0-delta spike
    # Y = np.delete(Y, [0, 1])
    # X = np.delete(X, [0, 1])

    # Histogram plot
    # ax.plot(X, Y, '-k', alpha=0.6)
    ax.plot(X, Y)
    ax.set_title(
        r'$\Delta$ - {} - $\gamma_n=\gamma=1, \alpha_n=\alpha=0.8$'.format(name), fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    # ax.set_xlim([0, 2.0])
    # ax.set_ylim([0, 1.5])
    plt.savefig(s.figures_folder + 'FIGURE1/' + 'D_{}.png'.format(name))
    plt.close()


# These are figures for individual countries - not needed for the paper.
# for name in s.countries:
#     G_B(name)
#     D(name)
# G_B('EU')
# D('EU')
# D('DE')

# This is the actual figure 1
# G_B_DE_and_EU()


# FIGURE 2 ---------------------------------------------------------------------------------------


def subplot_fancyname(resolution=100):
    # For DE

    # Data for subplot 1
    data_EU = np.empty(resolution)
    data_DE = np.empty(resolution)
    alpha_list = np.linspace(0, 1, resolution)

    for i, alpha in enumerate(alpha_list):
        Gw_EU, Gs_EU, L_EU, avg_L_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU = set_data(alpha, 1.0)
        country = 'DE'
        data_DE[i] = np.mean(G_B_n[s.country_dict[country]]) / \
            np.mean(L_EU[s.country_dict[country]])
        data_EU[i] = np.mean(G_B_EU) / np.mean(avg_L_EU)

    # Data for subplot 2
    n = 21
    alphas = np.linspace(0, 1.00, n)
    alpha = 1.00

    K_B_n_b0_q9 = np.empty(n)
    K_B_n_b0_q99 = np.empty(n)
    K_B_n_b0_q999 = np.empty(n)
    K_B_n_binf_q9 = np.empty(n)
    K_B_n_binf_q99 = np.empty(n)
    K_B_n_binf_q999 = np.empty(n)

    for i, alpha in enumerate(alphas):
        for beta in (0.00, np.inf):
            N = np.load(s.nodes_fullname.format(c='c', f='s', a=alpha, b=beta, g=1.00))
            balancing_DE = N.f.balancing[s.country_dict['DE']]
            if beta == 0.00:
                K_B_n_b0_q9[i] = t.quantile(0.9, balancing_DE)
                K_B_n_b0_q99[i] = t.quantile(0.99, balancing_DE)
                K_B_n_b0_q999[i] = t.quantile(0.999, balancing_DE)
            else:
                K_B_n_binf_q9[i] = t.quantile(0.9, balancing_DE)
                K_B_n_binf_q99[i] = t.quantile(0.99, balancing_DE)
                K_B_n_binf_q999[i] = t.quantile(0.999, balancing_DE)

    avg_L_DE = np.mean(L_EU[s.country_dict['DE']])
    K_B_n_b0_q9 /= avg_L_DE * 1000
    K_B_n_b0_q99 /= avg_L_DE * 1000
    K_B_n_b0_q999 /= avg_L_DE * 1000
    K_B_n_binf_q9 /= avg_L_DE * 1000
    K_B_n_binf_q99 /= avg_L_DE * 1000
    K_B_n_binf_q999 /= avg_L_DE * 1000

    # Plotting
    fig1, (ax1) = plt.subplots(1, 1)
    fig2, (ax2) = plt.subplots(1, 1)

    # Plot subplot 1
    ax1.plot(alpha_list, data_DE, '-r', label=r'$\beta^T = 0$')
    ax1.plot(alpha_list, data_EU, '-b', label=r'$\beta^T = \infty$')
    ax1.set_ylabel(r'$\frac{\left< G^B_n \right>}{\left< L_n \right> }$',
                   fontsize=15, rotation=0, labelpad=20)
    ax1.set_xlabel(r'$\alpha$', fontsize=20)

    # ax1.set_xlim([0, 1])
    # ax1.set_ylim([0, 0.6])
    ax1.legend(loc='lower left')

    # Plot subplot 2
    ax2.plot(alphas, K_B_n_b0_q9, ':r', label=r'$\beta^T=0, q=0.9$', alpha=0.75)
    ax2.plot(alphas, K_B_n_b0_q99, '-r', label=r'$\beta^T=0, q=0.99$', linewidth=2)
    ax2.plot(alphas, K_B_n_b0_q999, '--r', label=r'$\beta^T=0, q=0.999$', alpha=0.75)
    ax2.plot(alphas, K_B_n_binf_q9, ':b', label=r'$\beta^T=\infty, q=0.9$', alpha=0.75)
    ax2.plot(alphas, K_B_n_binf_q99, '-b', label=r'$\beta^T=\infty, q=0.99$', linewidth=2)
    ax2.plot(alphas, K_B_n_binf_q999, '--b', label=r'$\beta^T=\infty, q=0.999$', alpha=0.75)
    ax2.legend(loc='lower left', fontsize=15)
    ax2.set_ylabel(r'$\frac{K_n^B \left(q\right)}{\left<L_n\right>}$',
                   rotation=0, fontsize=20, labelpad=20)
    ax2.set_xlabel(r'$\alpha$', fontsize=20)

    fig1.savefig(s.figures_folder + 'FIGURE2/' + 'F21.pdf')
    fig2.savefig(s.figures_folder + 'FIGURE2/' + 'F22.pdf')
    # plt.show()
    plt.close()

subplot_fancyname()


# FIGURE 4 ----------------------------------------------------------------------------------------

def plot_figure4(summer_week=30, winter_week=60):

    # Extracting data
    Gw_EU, Gs_EU, L_EU, avg_L_EU, D_n, C_n, G_B_n, D_EU, C_EU, G_B_EU = set_data(0.8, 1.0)

    G_B_DE = G_B_n[s.country_dict['DE']]
    avg_L_DE = np.mean(L_EU[s.country_dict['DE']])
    data = G_B_DE / avg_L_DE

    # Finding two weeks

    summer_week = summer_week * 7 * 24
    winter_week = winter_week * 7 * 24
    two_weeks = 24 * 7 * 2
    data_summer = data[summer_week:summer_week + two_weeks]
    data_winter = data[winter_week:winter_week + two_weeks]

    x = [0, len(data_summer)]
    x_ticks = ['$t$', '$t$ + 2 weeks']
    # y = np.linspace(0, 1.25, 6)

    fig, (ax) = plt.subplots(1, 1)
    ax.plot(data_summer, 'y', label='Summer')
    ax.plot(data_winter, 'b', label='Winter')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    ax.axhline(0.7, color='grey', alpha=0.75, linestyle='--', label=r'$\approx 0.7$')
    ax.set_xticks(x)
    # ax.set_yticks(y)
    ax.set_xticklabels(x_ticks, ha="right", va="top", fontsize=20)
    ax.set_xlim(x)
    # ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\frac{G^B_n\left(t\right)}{\left<L_n\right>}$',
                  rotation=0, fontsize=20, labelpad=20)
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(s.figures_folder + 'FIGURE4/' + 'F4.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
    plt.close()

# plot_figure4(summer_week=52*1.5, winter_week=52*2)
# Figure1()
subplot_fancyname()
