import numpy as np
import matplotlib.pyplot as plt
import settings.settings as s
import settings.tools as t
import settings.prices as p
import seaborn as sns
from tqdm import tqdm
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')

"""
File to calculate objectives and costs
"""

# Variables to tune --------------------------------------------------------------------------------
# Network parameters
alpha = 0.80
gamma = 1.00
beta_b = 0.7

# Storage constraints
# Inf
KSPc = np.inf
KSPd = np.inf
eta_in = 1
eta_out = 1
KSE = np.inf

# Not inf
# eta_in = 0.5
# eta_out = 0.5
KSPc = 0.07
KSPd = 0.130
# KSE = 1.00
eta_in = p.prices_storage_bussar.EfficiencyCharge
eta_out = p.prices_storage_bussar.EfficiencyDischarge

# Loading the network time series ------------------------------------------------------------------
EU = [np.load(s.iset_folder + s.iset_prefix + s.files[country]) for country in range(30)]
L_EU = np.array([n['L'] for n in EU])
L_DE = L_EU[18]
avg_L_DE = np.mean(L_DE)
avg_L_DE *= 1000

N = np.load(s.nodes_fullname_inf.format(f='s', c='c', a=alpha, g=gamma, b=np.inf))
B = N.f.balancing[18]
B /= avg_L_DE
C = N.f.curtailment[18]
C /= avg_L_DE
nhours = N.f.nhours[18]
tt = range(len(B))

decimals_variables = 1
decimals_objectives = 2
decimals_LCOE = 3

# Solving the storage time series ------------------------------------------------------------------
# Unconstrained
(S_max0, S0, Bs0) = t.storage_size(B, beta_b)
# Constrained
(S_max1, S1, Bs1) = t.storage_size(B, beta_b,
                                   curtailment=C,
                                   eta_in=eta_in,
                                   eta_out=eta_out,
                                   charging_capacity=KSPc*eta_in,
                                   discharging_capacity=KSPd/eta_out,
                                   energy_capacity=KSE,
                                   )


# Amount of hours with non-served-energy -----------------------------------------------------------
E_NS = -(S0 - S1).clip(max=0)
E_NS[np.abs(E_NS) < 1e-10] = 0
E_NS_hours = np.count_nonzero(E_NS)

# Calculating storage objectives -------------------------------------------------------------------
# Unconstrained
(K_SE0, K_SPc0, K_SPd0) = t.get_objectives(S0, eta_in=1, eta_out=1)
# Constrained
(K_SE1, K_SPc1, K_SPd1) = t.get_objectives(S1, eta_in=eta_in, eta_out=eta_out)


# Calculating LCOE ---------------------------------------------------------------------------------
# Unconstrained
LCOE0 = t.calculate_costs(beta_b, np.sum(Bs0), K_SPc0, K_SPd0, K_SE0, L_DE,
                          p.prices_backup_leon, p.prices_storage_bussar)
(LCOE_BC_0, LCOE_BE_0, LCOE_SPCc_0, LCOE_SPCd_0, LCOE_SEC_0) = LCOE0
LCOE_B_0 = np.sum((LCOE_BC_0, LCOE_BE_0))
LCOE_S_0 = np.sum((LCOE_SPCc_0, LCOE_SPCd_0, LCOE_SEC_0))
# Constrained
LCOE1 = t.calculate_costs(beta_b, np.sum(Bs1), K_SPc1, K_SPd1, K_SE1, L_DE,
                          p.prices_backup_leon, p.prices_storage_bussar)
(LCOE_BC_1, LCOE_BE_1, LCOE_SPCc_1, LCOE_SPCd_1, LCOE_SEC_1) = LCOE1
LCOE_B_1 = np.sum((LCOE_BC_1, LCOE_BE_1))
LCOE_S_1 = np.sum((LCOE_SPCc_1, LCOE_SPCd_1, LCOE_SEC_1))
# Without storage
LCOE2 = t.calculate_costs(np.max(B), np.sum(B), 1, 1, 1, L_DE,
                          p.prices_backup_leon, p.prices_storage_bussar)
(LCOE_BC_2, LCOE_BE_2) = LCOE2[:2]
LCOE_B_2 = np.sum((LCOE_BC_2 + LCOE_BE_2))

# Printing results ---------------------------------------------------------------------------------
def print_objectives(K_B, BE, K_SPc, K_SPd, K_SE, eta_in, eta_out):
    s0 = 'BC      = {0:.{prec}f}'.format(K_B, prec=decimals_objectives)
    s1 = 'BE      = {0:.{prec}f}'.format(BE, prec=decimals_objectives)
    s2 = 'K_SPc   = {0:.{prec}f}'.format(K_SPc, prec=decimals_objectives)
    s3 = 'K_SPd   = {0:.{prec}f}'.format(K_SPd, prec=decimals_objectives)
    s4 = 'K_SE    = {0:.{prec}f}'.format(K_SE, prec=decimals_objectives)
    s5 = 'eta_in  = {0:.{prec}f}'.format(eta_in, prec=decimals_objectives)
    s6 = 'eta_out = {0:.{prec}f}'.format(eta_out, prec=decimals_objectives)
    s = '\n'.join((s0, s1, s2, s3, s4, s5, s6))
    print(s)
    return

def print_costs(LCOE_B, LCOE_S, LCOE_BC, LCOE_BE, LCOE_SPCc, LCOE_SPCd, LCOE_SEC):
    s0 = 'LCOE Total = {0:.{prec}f}'.format(LCOE_B + LCOE_S, prec=decimals_LCOE)
    s1 = 'LCOE B     = {0:.{prec}f}'.format(LCOE_B, prec=decimals_LCOE)
    s2 = 'LCOE BC    = {0:.{prec}f}'.format(LCOE_BC, prec=decimals_LCOE)
    s3 = 'LCOE BE    = {0:.{prec}f}'.format(LCOE_BE, prec=decimals_LCOE)
    s4 = 'LCOE S     = {0:.{prec}f}'.format(LCOE_S, prec=decimals_LCOE)
    s5 = 'LCOE SPCc  = {0:.{prec}f}'.format(LCOE_SPCc, prec=decimals_LCOE)
    s6 = 'LCOE SPCd  = {0:.{prec}f}'.format(LCOE_SPCd, prec=decimals_LCOE)
    s7 = 'LCOE SEC   = {0:.{prec}f}'.format(LCOE_SEC, prec=decimals_LCOE)
    s = '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7))
    print(s)
    return

def print_all_objectives():
    print('----------------------------------------------------------------------------------------')
    print('Network parameters: '.upper())
    print('alpha: {0:.{prec}f}, beta^B: {1:.{prec}f}, gamma: {2:.{prec}f}'.format(alpha, beta_b,
                                                                                  gamma, prec=decimals_variables))
    print('\n-------- Objectives: --------'.upper())
    print('[<L_n>]')
    print('UNCONSTRAINED storage: ')
    print_objectives(beta_b, np.sum(Bs0), K_SPc0, K_SPd0, K_SE0, 1, 1)
    print('\nCONSTRAINED storage: ')
    print_objectives(beta_b, np.sum(Bs1), K_SPc1, K_SPd1, K_SE1, eta_in, eta_out)
    print('\nNon-served energy: {0:.2f}\nNon-served hours: {1}/{2} = {3:.4f}'.format(np.sum(E_NS),
        E_NS_hours, nhours, E_NS_hours/nhours))
    print('\nWITHOUT storage')
    print_objectives(np.max(B), np.sum(B), 0, 0, 0, 0, 0)
    return

def print_all_costs():
    print('\n-------- LCOE: --------')
    print('[Euros/MWh]')
    print('UNCONSTRAINED storage:')
    print_costs(LCOE_B_0, LCOE_S_0, LCOE_BC_0, LCOE_BE_0, LCOE_SPCc_0, LCOE_SPCd_0, LCOE_SEC_0)
    print('\nCONSTRAINED storage:')
    print_costs(LCOE_B_1, LCOE_S_1, LCOE_BC_1, LCOE_BE_1, LCOE_SPCc_1, LCOE_SPCd_1, LCOE_SEC_1)
    print('\nWITHOUT storage:')
    print_costs(LCOE_B_2, 0, LCOE_BC_2, LCOE_BE_2, 0, 0, 0)
    print('----------------------------------------------------------------------------------------')
    return


# Plotting timeseries ------------------------------------------------------------------------------
def plot_timeseries():
    linew = 1
    fig0, (ax0) = plt.subplots(1, 1)
    ax0.plot(range(len(S0)), S0)
    ax0.plot(range(len(S1)), S1)
    ax0.plot(range(len(B)), B, label=r'$G_n^B$')
    ax0.plot(range(len(Bs0)), Bs0, label=r'Unconstrained $G_n^{{BS}}$')
    ax0.plot(range(len(Bs1)), Bs1, label=r'Constrained $G_n^{{BS}}$')
    ax0.set_xlim(tt[0], tt[-1])
# Strings for labels
    s01 = r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.2f}, \quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.2f}$'
    s02 = r'$C = {2}$'
    s03 = r'$\mathcal{{K}}^{{SPc}}={3}, \mathcal{{K}}^{{SPd}}={4}, \mathcal{{K}}^{{SE}}={5}$'
    s0all = '\n'.join((s01, s02, s03))
# Making the string pretty with infty in latex
    ax0.lines[0].set_label(s0all.format(1, 1, 0, '\infty', '\infty', '\infty'))
    ax0.lines[1].set_label(s0all.format(eta_in, eta_out, 'C', KSPc if KSPc != np.inf else '\infty',
                                                            KSPd if KSPd != np.inf else '\infty',
                                                            KSE if KSE != np.inf else '\infty'))
    ax0.legend(loc='lower center', fontsize=12)
# Strings for objectives
    s11 = r'$\beta^B = {0}, \quad \alpha = {1}, \quad \gamma = {2}$'.format(beta_b, alpha, gamma)
    s12 = '\nUnconstrained: '
    s13 = r'$\mathcal{{K}}^{{SPc}}={0:.2f},\quad \mathcal{{K}}^{{SPd}}={1:.2f}$'
    s17 = r'$\mathcal{{K}}^{{SE}} = {0:.2f}$'
    s14 = '\nConstrained: '
    s15 = r'$E^{{NS}} = {0:.2f}$'.format(np.sum(E_NS))
    s181 = '\n Differences: '
    s182 = r'$\Delta \mathcal{{K}}^{{SPc}} = {0:.2f}$'.format(K_SPc1 - K_SPc0)
    s183 = r'$\Delta \mathcal{{K}}^{{SPd}} = {0:.2f}$'.format(K_SPd1 - K_SPd0)
    s184 = r'$\Delta \mathcal{{K}}^{{SE}} = {0:.2f}$'.format(K_SE1 - K_SE0)
    s16 = '\nNon-served hours: \n${0}/{1} = {2:.5f}$'.format(E_NS_hours, nhours, E_NS_hours/nhours)
# Textbox for strings
    s1all = '\n'.join((s11, s12,
                       s13.format(K_SPc0, K_SPd0),
                       s17.format(K_SE0), s14,
                       s13.format(K_SPc1, K_SPd1),
                       s17.format(K_SE1),
                       s181, s182, s183, s184,
                       s16, s15))
    anchored_text1 = AnchoredText(s1all, loc=3, frameon=False)
    ax0.add_artist(anchored_text1)

    ax0.set_xlabel(r'$t [y]$')
    ax0.set_ylabel(r'$S(t)\left[\langle L_n \rangle \right]$')
    ax0.set_xlim(tt[0], tt[-1])
    ax0.set_xticks(np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[-1], 9))
    ax0.set_xticklabels(range(0, 9))
    fig0.savefig(s.figures_folder + 'storage.pdf')

    plt.show()
    plt.close('all')


def plot_bar():
    colors = sns.color_palette()
    index = np.arange(3)
    width = 0.5
    BC = np.array([LCOE_BC_0, LCOE_BC_1, LCOE_BC_2])
    BE = np.array([LCOE_BE_0, LCOE_BE_1, LCOE_BE_2])
    SPCc = np.array([LCOE_SPCc_0, LCOE_SPCc_1, 0])
    SPCd = np.array([LCOE_SPCd_0, LCOE_SPCd_1, 0])
    SEC = np.array([LCOE_SEC_0, LCOE_SEC_1, 0])
    fig1, (ax1) = plt.subplots(1, 1)
    p1 = ax1.bar(index, BC, width, color=colors[0])
    p2 = ax1.bar(index, BE, width, bottom=BC, color=colors[1])
    p3 = ax1.bar(index, SPCc, width, bottom=BC+BE, color=colors[2])
    p4 = ax1.bar(index, SPCd, width, bottom=BC+BE+SPCc, color=colors[3])
    p5 = ax1.bar(index, SEC, width, bottom=BC+BE+SPCc+SPCd, color=colors[4])
    ax1.set_ylabel(r'$€/MWh$')
    ax1.xaxis.grid(False)
    plt.xticks(index + width/2., ('Unconstrained Storage', 'Constrained Storage', 'Without Storage'))
    labels = ('BC', 'BE', 'SPCc', 'SPCd', 'SEC')
    ax1.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), labels, loc='best')
    s01 = (r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.{prec}f},'
            '\quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.{prec}f}$'.format(eta_in, eta_out, prec=2))
    s02 = (r'$\mathcal{{K}}^{{SPc}}={0:.{prec}f}, \mathcal{{K}}^{{SPd}}={1:.{prec}f}, '
            '\mathcal{{K}}^{{SE}}={2:.{prec}f}$'.format(K_SPc1, K_SPd1, K_SE1, prec=4))
    s03 = r'$E^{{NS}} = {0:.{prec}f}, \quad NS: {1:.0f}/{2:.0f}={3:.{prec}f}$'.format(np.sum(E_NS),
            E_NS_hours, nhours, E_NS_hours/nhours, prec=4)
    s0all = '\n'.join((s01, s02, s03))
    anchored_text0 = AnchoredText(s0all, loc=9, frameon=False)
    ax1.add_artist(anchored_text0)
    plt.show()
    fig1.savefig(s.figures_folder + 'stacked_LCOE.pdf')
    plt.close()


print_all_objectives()
print_all_costs()
# plot_timeseries()
plot_bar()
