import numpy as np
import matplotlib.pyplot as plt
import settings.settings as s
import settings.tools as t
import settings.prices as p
from tqdm import tqdm
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')

"""
"""

# Variables to tune --------------------------------------------------------------------------------
alpha = 0.80
gamma = 1.00
beta_b = 0.7

KSPc = np.inf
KSPd = np.inf
eta_in = 1
eta_out = 1
# eta_in = 0.5
# eta_out = 0.5
KSPc = 0.01
KSPd = 0.15
KSE = np.inf
# KSE = 3.00
eta_in = p.prices_storage_david.EfficiencyCharge
eta_out = p.prices_storage_david.EfficiencyDischarge

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

# Printing results ---------------------------------------------------------------------------------
def print_costs(LCOE_B, LCOE_S, LCOE_BC, LCOE_BE, LCOE_SPCc, LCOE_SPCd, LCOE_SEC, title=''):
    s0 = 'LCOE Total = {0:.2f}'.format(LCOE_B + LCOE_S)
    s1 = 'LCOE B     = {0:.2f}'.format(LCOE_B)
    s2 = 'LCOE BC    = {0:.2f}'.format(LCOE_BC)
    s3 = 'LCOE BE    = {0:.2f}'.format(LCOE_BE)
    s4 = 'LCOE S     = {0:.2f}'.format(LCOE_S)
    s5 = 'LCOE SPCc  = {0:.2f}'.format(LCOE_SPCc)
    s6 = 'LCOE SPCd  = {0:.2f}'.format(LCOE_SPCd)
    s7 = 'LCOE SEC   = {0:.2f}'.format(LCOE_SEC)
    s = '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7))
    print(s)
    return

def print_objectives(K_B, BE, K_SPc, K_SPd, K_SE, eta_in, eta_out):
    s0 = 'K_B     = {0:.4f}'.format(K_B)
    s1 = 'BE      = {0:.4f}'.format(BE)
    s2 = 'K_SPc   = {0:.4f}'.format(K_SPc)
    s3 = 'K_SPd   = {0:.4f}'.format(K_SPd)
    s4 = 'K_SE    = {0:.4f}'.format(K_SE)
    s5 = 'eta_in  = {0:.4f}'.format(eta_in)
    s6 = 'eta_out = {0:.4f}'.format(eta_out)
    s = '\n'.join((s0, s1, s2, s3, s4, s5, s6))
    print(s)
    return

def print_all_objectives():
    print('Network parameters: ')
    print('alpha: {0:.2f}, beta^B: {1:.2f}, gamma: {2:.2f}'.format(alpha, beta_b, gamma))
    print('\n#### Objectives: ####')
    print('Unconstrained: ')
    print_objectives(beta_b, np.sum(Bs0), K_SPc0, K_SPd0, K_SE0, 1, 1)
    print('\nConstrained: ')
    print_objectives(beta_b, np.sum(Bs1), K_SPc1, K_SPd1, K_SE1, eta_in, eta_out)
    print('\nNon-served energy: {0:.2f}\nNon-served hours: {1}'.format(np.sum(E_NS), E_NS_hours))
    return

def print_all_costs():
    print('\n#### LCOE: ####')
    print('Unconstrained:')
    print_costs(LCOE_B_0, LCOE_S_0, LCOE_BC_0, LCOE_BE_0, LCOE_SPCc_0, LCOE_SPCd_0, LCOE_SEC_0)
    print('\nConstrained:')
    print_costs(LCOE_B_1, LCOE_S_1, LCOE_BC_1, LCOE_BE_1, LCOE_SPCc_1, LCOE_SPCd_1, LCOE_SEC_1)
    return

# Plotting timeseries ------------------------------------------------------------------------------
def plot_timeseries():
    linew = 1
    fig0, (ax0) = plt.subplots(1, 1)
    ax0.plot(range(len(S0)), S0)
    ax0.plot(range(len(S1)), S1)
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
    s16 = '\nNon-served hours: \n${0}/{1} = {2:.4f}$'.format(E_NS_hours, nhours, E_NS_hours/nhours)
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

print_all_objectives()
print_all_costs()
# plot_timeseries()
