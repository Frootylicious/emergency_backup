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

class LCOE_Objectives():
    '''
    Class to calculate a storage with corresponding objectives and LCOE.
    '''
    def __init__(self):
        # Decimals.
        self.decimals_variables = 1
        self.decimals_objectives = 4
        self.decimals_LCOE = 4
        self.colors = sns.color_palette()

        # Do stuff.
        self.do_all()

    def do_all(self):
        self.set_all()
        self.calculate_all()
#         self.print_all()
        self.plot_all()

    def set_all(self):
        self.set_network_parameters()
        self.set_storage_constraints(
                                     KSPc=0.0014, # OPTIMAL VALUES
#                                      KSPc=0.0012, 
                                     KSPd=0.17, # OPTIMAL VALUES
#                                      KSE=2.0,
                                     eta_in=p.prices_storage_bussar.EfficiencyCharge,
                                     eta_out=p.prices_storage_bussar.EfficiencyDischarge,
                                     add_curtailment=True,
                                    )

    def calculate_all(self):
        self.load_network()
        self.calculate_storage()
        self.calculate_E_NS()
        self.get_objectives()
        self.calculate_LCOE()

    def print_all(self):
        self.print_objectives()
        self.print_costs()

    def plot_all(self):
        # self.plot_timeseries()
#          self.plot_limited_timeseries()
#         self.plot_bar()
#         self.plot_single_bar()
        return

    # Setting parameters for network.
    def set_network_parameters(self, a=0.80, g=1.00, b=0.70):
        self.alpha = a
        self.gamma = g
        self.beta_b = b

    # Setting constraints for storage.
    def set_storage_constraints(self,
                                KSE=np.inf,
                                KSPc=np.inf,
                                KSPd=np.inf,
                                eta_in=1,
                                eta_out=1,
                                add_curtailment=False,
                                ):
        self.KSPc = KSPc
        self.KSPd = KSPd
        self.eta_in = eta_in
        self.eta_out = eta_out
        self.add_curtailment = add_curtailment
        self.KSE = KSE

    # Loading the solved network and extracting the time series in unit of mean load. --------------
    def load_network(self):
        self.EU = [np.load(s.iset_folder + s.iset_prefix + s.files[country]) for country in range(30)]
        self.L_EU = np.array([n['L'] for n in self.EU])
        self.L_DE = self.L_EU[18]
        self.avg_L_DE = np.mean(self.L_DE)
        self.avg_L_DE *= 1000

        self.N = np.load(s.nodes_fullname_inf.format(f='s', c='c', a=self.alpha, g=self.gamma, b=np.inf))
        self.B = self.N.f.balancing[18]
        self.B /= self.avg_L_DE
        # Backup capacities at 99, 99.9 and 99.99 % quantiles.
        self.cap99 = t.quantile_old(0.99, self.B)
        self.cap999 = t.quantile_old(0.999, self.B)
        self.cap9999 = t.quantile_old(0.9999, self.B)

        self.B_99 = self.B.clip(max=self.cap99)
        self.B_999 = self.B.clip(max=self.cap999)
        self.B_9999 = self.B.clip(max=self.cap9999)
        self.C = self.N.f.curtailment[18]
        self.C /= self.avg_L_DE
        self.nhours = self.N.f.nhours[18]
        self.tt = range(len(self.B))

    # Calculate storage and objectives -------------------------------------------------------------
    def calculate_storage(self):
        # Unconstrained Storage
        (self.S_max0, self.S0, self.Bs0) = t.storage_size(self.B, self.beta_b)
        # Constrained Storage
        (self.S_max1, self.S1,self. Bs1) = t.storage_size(self.B, self.beta_b,
                                                          curtailment=self.C if self.add_curtailment else np.array(0),
                                                          eta_in=self.eta_in,
                                                          eta_out=self.eta_out,
                                                          charging_capacity=self.KSPc*self.eta_in,
                                                          discharging_capacity=self.KSPd/self.eta_out,
                                                          energy_capacity=self.KSE,
                                                          )

        # Find number of non served hours as a result of the constraints compared to the
        # unconstrained storage.
#         self.E_NS = -(self.S0 - self.S1).clip(max=0)
#         self.E_NS[np.abs(self.E_NS) < 1e-10] = 0
#         self.E_NS_hours = np.count_nonzero(self.E_NS)

    def calculate_E_NS(self):
        self.E_NS, self.E_NS_hours = t.get_non_served_storage_energy(self.S0, self.S1,
                self.eta_in, self.eta_out)

#         diff0 = np.append(0, np.diff(self.S0)).clip(max=0)
#         diff1 = np.append(0, np.diff(self.S1)).clip(max=0)
#         diff1[diff1 < 0] *= self.eta_out  # Obsolete since we're only interested in discharging events
#         diff1[diff1 > 0] /= self.eta_in
#         self.diff0 = diff0
#         self.diff1 = diff1
#         self.E_NS_timeseries = -(diff0 - diff1).clip(max=0)
#         self.E_NS_timeseries[np.abs(self.E_NS_timeseries) < 1e-10] = 0
#         self.E_NS_hours = np.count_nonzero(self.E_NS_timeseries)
#         self.E_NS = self.E_NS_timeseries



    def get_objectives(self):
        # Unconstrained
        (self.K_SE0, self.K_SPc0, self.K_SPd0) = t.get_objectives(self.S0, eta_in=1, eta_out=1)
        # Constrained
        (self.K_SE1, self.K_SPc1, self.K_SPd1) = t.get_objectives(self.S1, eta_in=self.eta_in, eta_out=self.eta_out)

    # Calculate prices -----------------------------------------------------------------------------
    def calculate_LCOE(self):
        # Unconstrained
        self.LCOE0 = t.calculate_costs(self.beta_b, np.sum(self.Bs0), self.K_SPc0, self.K_SPd0, self.K_SE0, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_0, self.LCOE_BE_0, self.LCOE_SPCc_0, self.LCOE_SPCd_0, self.LCOE_SEC_0) = self.LCOE0
        self.LCOE_B_0 = np.sum((self.LCOE_BC_0, self.LCOE_BE_0))
        self.LCOE_S_0 = np.sum((self.LCOE_SPCc_0, self.LCOE_SPCd_0, self.LCOE_SEC_0))
        # Constrained
        self.LCOE1 = t.calculate_costs(self.beta_b, np.sum(self.Bs1), self.K_SPc1, self.K_SPd1, self.K_SE1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_1, self.LCOE_BE_1, self.LCOE_SPCc_1, self.LCOE_SPCd_1, self.LCOE_SEC_1) = self.LCOE1
        self.LCOE_B_1 = np.sum((self.LCOE_BC_1, self.LCOE_BE_1))
        self.LCOE_S_1 = np.sum((self.LCOE_SPCc_1, self.LCOE_SPCd_1, self.LCOE_SEC_1))
        # Without storage - 100 % coverage.
        self.LCOE2 = t.calculate_costs(np.max(self.B), np.sum(self.B), 1, 1, 1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        sizes = [np.sum(self.Bs1), self.beta_b, self.K_SE1, self.K_SPc1, self.K_SPd1]
        (self.LCOE_BC_2, self.LCOE_BE_2) = self.LCOE2[:2]
        self.LCOE_B_2 = np.sum((self.LCOE_BC_2 + self.LCOE_BE_2))

        # Without storage - 99 % coverage.
        self.LCOE_99 = t.calculate_costs(np.max(self.B_99), np.sum(self.B_99), 1, 1, 1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_99, self.LCOE_BE_99) = self.LCOE_99[:2]
        self.LCOE_B_99 = np.sum((self.LCOE_BC_99 + self.LCOE_BE_99))
        # Without storage - 99.9 % coverage.
        self.LCOE_999 = t.calculate_costs(np.max(self.B_999), np.sum(self.B_999), 1, 1, 1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_999, self.LCOE_BE_999) = self.LCOE_999[:2]
        self.LCOE_B_999 = np.sum((self.LCOE_BC_999 + self.LCOE_BE_999))
        # Without storage - 99.99 % coverage.
        self.LCOE_9999 = t.calculate_costs(np.max(self.B_9999), np.sum(self.B_9999), 1, 1, 1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_9999, self.LCOE_BE_9999) = self.LCOE_9999[:2]
        self.LCOE_B_9999 = np.sum((self.LCOE_BC_9999 + self.LCOE_BE_9999))

    # Printing results -----------------------------------------------------------------------------
    def print_objectives(self, only_constrained=False):
        def _get_objective_string(K_B, BE, K_SPc, K_SPd, K_SE, eta_in, eta_out):
            s0 = 'BC      = {0:.{prec}f}'.format(K_B, prec=self.decimals_objectives)
            s1 = 'BE      = {0:.{prec}f}'.format(BE, prec=self.decimals_objectives)
            s2 = 'K_SPc   = {0:.{prec}f}'.format(K_SPc, prec=self.decimals_objectives)
            s3 = 'K_SPd   = {0:.{prec}f}'.format(K_SPd, prec=self.decimals_objectives)
            s4 = 'K_SE    = {0:.{prec}f}'.format(K_SE, prec=self.decimals_objectives)
            s5 = 'eta_in  = {0:.{prec}f}'.format(eta_in, prec=self.decimals_objectives)
            s6 = 'eta_out = {0:.{prec}f}'.format(eta_out, prec=self.decimals_objectives)
            s = '\n'.join((s0, s1, s2, s3, s4, s5, s6))
            return(s)

        s0 = '----- Network Parameters -----'
        s1 = 'alpha: {0:.{prec}f} | beta^B: {1:.{prec}f} | gamma: {2:.{prec}f}'.format(self.alpha,
                                                                                     self.beta_b,
                                                                                     self.gamma,
                                                                                     prec=self.decimals_variables)
        s2 = '--------- OBJECTIVES ---------'
        s3 = '[<L_n>]'
        s4 = 'UNCONSTRAINED storage: -------'
        s5 = _get_objective_string(self.beta_b, np.sum(self.Bs0), self.K_SPc0, self.K_SPd0, self.K_SE0, 1, 1)
        s6 = 'CONSTRAINED storage: ---------'
        s7 = _get_objective_string(self.beta_b, np.sum(self.Bs1), self.K_SPc1, self.K_SPd1, self.K_SE1, self.eta_in, self.eta_out)
        s8 = '\nE^NS    = {0:.2f}\nhours: {1}/{2} \n        = {3:.4f}\n'.format(np.sum(self.E_NS),
                                                                                      self.E_NS_hours,
                                                                                      self.nhours,
                                                                                      self.E_NS_hours/self.nhours)
        s9 = 'NO storage: 100 % ------------'
        s10 = _get_objective_string(np.max(self.B), np.sum(self.B), 0, 0, 0, 0, 0)
        s11 = 'NO storage: 99 % -------------'
        s12 = _get_objective_string(np.max(self.B_99), np.sum(self.B_99), 0, 0, 0, 0, 0)
        s13 = 'NO storage: 99.9 % -----------'
        s14 = _get_objective_string(np.max(self.B_999), np.sum(self.B_999), 0, 0, 0, 0, 0)
        s15 = 'NO storage: 99.99 % ----------'
        s16 = _get_objective_string(np.max(self.B_9999), np.sum(self.B_9999), 0, 0, 0, 0, 0)
        if only_constrained:
            s = '\n'.join(('\n', s0, s1, s2, s3, s6, s7))
        else:
            s = '\n'.join(('\n', s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16))
        print(s)

    def print_costs(self, only_constrained=False):
        def _get_costs_string(LCOE_B, LCOE_S, LCOE_BC, LCOE_BE, LCOE_SPCc, LCOE_SPCd, LCOE_SEC):
            s0 = 'LCOE Total = {0:.{prec}f}'.format(LCOE_B + LCOE_S, prec=self.decimals_LCOE)
            s1 = 'LCOE B     = {0:.{prec}f}'.format(LCOE_B, prec=self.decimals_LCOE)
            s2 = 'LCOE BC    = {0:.{prec}f}'.format(LCOE_BC, prec=self.decimals_LCOE)
            s3 = 'LCOE BE    = {0:.{prec}f}'.format(LCOE_BE, prec=self.decimals_LCOE)
            s4 = 'LCOE S     = {0:.{prec}f}'.format(LCOE_S, prec=self.decimals_LCOE)
            s5 = 'LCOE SPCc  = {0:.{prec}f}'.format(LCOE_SPCc, prec=self.decimals_LCOE)
            s6 = 'LCOE SPCd  = {0:.{prec}f}'.format(LCOE_SPCd, prec=self.decimals_LCOE)
            s7 = 'LCOE SEC   = {0:.{prec}f}'.format(LCOE_SEC, prec=self.decimals_LCOE)
            s = '\n'.join((s0, s1, s2, s3, s4, s5, s6, s7))
            return(s)

        s0 = '------------ LCOE ------------'
        s1 = '[Euros/MWh]'
        s2 = 'UNCONSTRAINED storage: -------'
        s3 = _get_costs_string(self.LCOE_B_0, self.LCOE_S_0,
                               self.LCOE_BC_0, self.LCOE_BE_0,
                               self.LCOE_SPCc_0, self.LCOE_SPCd_0, self.LCOE_SEC_0)
        s4 = 'CONSTRAINED storage: ---------'
        s5 = _get_costs_string(self.LCOE_B_1, self.LCOE_S_1,
                               self.LCOE_BC_1, self.LCOE_BE_1,
                               self.LCOE_SPCc_1, self.LCOE_SPCd_1, self.LCOE_SEC_1)
        s6 = 'NO storage 100 % -------------'
        s7 = _get_costs_string(self.LCOE_B_2, 0,
                               self.LCOE_BC_2, self.LCOE_BE_2,
                               0, 0, 0)
        s8 = 'NO storage 99 % --------------'
        s9 = _get_costs_string(self.LCOE_B_99, 0,
                               self.LCOE_BC_99, self.LCOE_BE_99,
                               0, 0, 0)
        s10 = 'NO storage 99.9 % ------------'
        s11 = _get_costs_string(self.LCOE_B_999, 0,
                               self.LCOE_BC_999, self.LCOE_BE_999,
                               0, 0, 0)
        s12 = 'NO storage 99.99 % -----------'
        s13 = _get_costs_string(self.LCOE_B_9999, 0,
                               self.LCOE_BC_9999, self.LCOE_BE_9999,
                               0, 0, 0)
        if only_constrained:
            s = '\n'.join(('\n', s0, s1, s4, s5))
        else:
            s = '\n'.join(('\n', s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13))
        print(s)


    # Plotting ------ ------------------------------------------------------------------------------
    def plot_timeseries(self):
        linew = 1
        fig, (ax) = plt.subplots(1, 1)
        ax.plot(range(len(self.S0)), self.S0)
        ax.plot(range(len(self.S1)), self.S1)
        ax.plot(range(len(self.B)), self.B, label=r'$G_n^B$')
        ax.plot(range(len(self.Bs0)), self.Bs0, label=r'Unconstrained $G_n^{{BS}}$')
        ax.plot(range(len(self.Bs1)), self.Bs1, label=r'Constrained $G_n^{{BS}}$')
        ax.plot(range(len(self.C)), self.C, label=r'C')
        ax.plot(range(len(self.E_NS)), self.E_NS, 'k', label='Non-served storage energy $E^{NS, S}$')

    # Strings for labels
        s01 = r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.2f}, \quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.2f}$'
        s02 = r'$C = {2}$'
        s03 = r'$\mathcal{{K}}^{{SPc}}={3}, \mathcal{{K}}^{{SPd}}={4}, \mathcal{{K}}^{{SE}}={5}$'
        s0all = '\n'.join((s01, s02, s03))
    # Making the string pretty with infty in latex
        ax.lines[0].set_label(s0all.format(1, 1, 0, '\infty', '\infty', '\infty'))
        ax.lines[1].set_label(s0all.format(self.eta_in, self.eta_out, 'C', self.KSPc if self.KSPc != np.inf else '\infty',
                                                                self.KSPd if self.KSPd != np.inf else '\infty',
                                                                self.KSE if self.KSE != np.inf else '\infty'))
        ax.legend(loc='lower center', fontsize=12)
    # Strings for objectives
        s11 = r'$\beta^B = {0}, \quad \alpha = {1}, \quad \gamma = {2}$'.format(self.beta_b, self.alpha, self.gamma)
        s12 = '\nUnconstrained: '
        s13 = r'$\mathcal{{K}}^{{SPc}}={0:.3f},\quad \mathcal{{K}}^{{SPd}}={1:.3f}$'
        s17 = r'$\mathcal{{K}}^{{SE}} = {0:.3f}$'
        s14 = '\nConstrained: '
        s15 = r'$E^{{NS}} = {0:.2f}$'.format(np.sum(self.E_NS))
        s181 = '\n Differences: '
        s182 = r'$\Delta \mathcal{{K}}^{{SPc}} = {0:.3f}$'.format(self.K_SPc1 - self.K_SPc0)
        s183 = r'$\Delta \mathcal{{K}}^{{SPd}} = {0:.3f}$'.format(self.K_SPd1 - self.K_SPd0)
        s184 = r'$\Delta \mathcal{{K}}^{{SE}} = {0:.3f}$'.format(self.K_SE1 - self.K_SE0)
        s16 = '\nNon-served hours: \n${0}/{1} = {2:.5f}$'.format(self.E_NS_hours, self.nhours, self.E_NS_hours/self.nhours)
    # Textbox for strings
        s1all = '\n'.join((s11, s12,
                           s13.format(self.K_SPc0, self.K_SPd0),
                           s17.format(self.K_SE0), s14,
                           s13.format(self.K_SPc1, self.K_SPd1),
                           s17.format(self.K_SE1),
                                      s181, s182, s183, s184,
                                      s16, s15))
        anchored_text1 = AnchoredText(s1all, loc=3, frameon=False)
        ax.add_artist(anchored_text1)

        ax.set_xlabel(r'$t [y]$')
        ax.set_ylabel(r'$S(t)\left[\langle L_n \rangle \right]$')
        ax.set_xlim(self.tt[0], self.tt[-1])
        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[-1], 9))
        ax.set_xticklabels(range(0, 9))
        fig.savefig(s.figures_folder + 'storage.pdf')

        plt.show()
        plt.close('all')

    def plot_curtailment_timeseries(self):
        linew = 1
        fig, (ax) = plt.subplots(1, 1)
        ax.plot(range(len(self.S0)), self.S0, label=r'$S_n$ without $C$')
        ax.plot(range(len(self.S1)), self.S1, label=r'$S_n$ with $C$')
        ax.plot(range(len(self.Bs0)), self.Bs0, label=r'$G_n^{{BS}}$ without $C$')
        ax.plot(range(len(self.Bs1)), self.Bs1, label=r'$G_n^{{BS}}$ with $C$')
        ax.plot(range(len(self.C)), self.C, label=r'C')

        ax.set_xlabel(r'$t [h]$')
        ax.set_ylabel(r'$\left[\langle L_n \rangle \right]$')
        ax.set_xlim([53490, 53501])
        ax.set_ylim([-3, 1])
        ax.legend(loc='best', fontsize=12)
        fig.savefig(s.figures_folder + 'curtailment_timeseries.pdf')

        plt.show()
        plt.close('all')

    def plot_efficiencies_timeseries(self):
        linew = 1
        fig, (ax) = plt.subplots(1, 1)
        ax.plot(range(len(self.S0)), self.S0, label=r'$S_n$ without $\eta$')
        ax.plot(range(len(self.S1)), self.S1, label=r'$S_n$ with $\eta$')
        ax.plot(range(len(self.Bs0)), self.Bs0, label=r'$G_n^{{BS}}$ without $\eta$')
        ax.plot(range(len(self.Bs1)), self.Bs1, label=r'$G_n^{{BS}}$ with $\eta$')

        ax.set_xlabel(r'$t [h]$')
        ax.set_ylabel(r'$\left[\langle L_n \rangle \right]$')
        ax.set_xlim([53360, 53485])
        ax.set_ylim([-12, 1])
        ax.legend(loc='best', fontsize=12)
        fig.savefig(s.figures_folder + 'efficiencies_timeseries.pdf')

        plt.show()
        plt.close('all')

    def plot_limited_timeseries(self):
        colors = sns.color_palette()
        linew = 1
        xlim = [365*24+2*7*24+15, 365*24+3*7*24+12]
        fig, (ax) = plt.subplots(1, 1, figsize=s.figure_size)
        length = len(self.S1)
        rlength = range(length)
        ax.plot(rlength, self.S0, label='Unconstrained $S_n(t)$',color=colors[0])
        ax.plot(rlength, self.S1, label='Constrained $S_n(t)$',color=colors[1])
        ax.plot(rlength, self.B, 'gray', label=r'$G_n^B(t)$')
        ax.plot(rlength, self.Bs0, label=r'Unconstrained $G_n^{BS}(t)$',color=colors[2])
        ax.plot(rlength, self.Bs1, label=r'Constrained $G_n^{BS}(t)$',color=colors[3])
#         ax.plot(rlength, self.S1 - self.S0)
        ax.plot(rlength, [self.beta_b]*length, label='$\mathcal{{K}}^B={}$'.format(self.beta_b),color=colors[4])
#         ax.plot(rlength, [-0.5]*length, label=r'$\mathcal{K}^{SE}_{\max}=-0.5$', color=colors[5])

        ax.set_xlim(xlim)
        ax.set_ylim([-1.5, 1.5])

        ax.set_xlabel(r'$t\, [h]$')
        ax.set_ylabel(r'$\left[\langle L_n \rangle \right]$')
        ax.legend(loc='upper center', ncol=3, fontsize=12)


#         name = 'timeseries_limited_constraints_KSE'
#         name = 'timeseries_limited_constraints_KSPc'
        name = 'timeseries_limited_constraints_KSPd'


        fig.savefig(s.figures_folder + name + '.pdf')

        plt.show()
        plt.close('all')

    def plot_single_bar(self):
        colors = sns.color_palette()
        fig, (ax) = plt.subplots(1, 1)
        sizes = [
                 self.LCOE_BC_1,
                 self.LCOE_BE_1,
                 self.LCOE_SPCc_1,
                 self.LCOE_SPCd_1,
                 self.LCOE_SEC_1,
                 ]
        labels = ['$\mathcal{K}^{B}$', '$E^B$', '$\mathcal{K}^{SPc}$', '$\mathcal{K}^{SPd}$', '$\mathcal{K}^{SE}$']

#         patches, texts = plt.pie(sizes, colors=colors)
#         plt.legend(patches, labels, loc="best")
        plt.pie(sizes,              # data
                labels=labels,      # slice labels
                colors=colors,      # array of colours
                autopct='%1.2f%%',  # print the values inside the wedges
                startangle=90,       # starting angle
	        labeldistance=0.9,
        )

        plt.axis('equal')
        plt.show()
        fig.savefig(s.figures_folder + 'single_bar.pdf')
        plt.close('all')


    def plot_bar(self, show_numbers=True):
        colors = sns.color_palette()
        index = np.arange(6)
        width = 0.5
        # Sequence: 100, 99.99, 99.9, 99, constrained, unconstrained.
        # 2, 9999, 999, 99, 0, 1
        BC = np.array([
                       self.LCOE_BC_2,
                       self.LCOE_BC_9999,
                       self.LCOE_BC_999,
                       self.LCOE_BC_99,
                       self.LCOE_BC_0,
                       self.LCOE_BC_1,
                       ])
        BE = np.array([
                       self.LCOE_BE_2,
                       self.LCOE_BE_9999,
                       self.LCOE_BE_999,
                       self.LCOE_BE_99,
                       self.LCOE_BE_0,
                       self.LCOE_BE_1,
                       ])
        SPCc = np.array([0, 0, 0, 0, self.LCOE_SPCc_0, self.LCOE_SPCc_1])
        SPCd = np.array([0, 0, 0, 0, self.LCOE_SPCd_0, self.LCOE_SPCd_1])
        SEC = np.array([0, 0, 0, 0, self.LCOE_SEC_0, self.LCOE_SEC_1])
        fig, (ax) = plt.subplots(1, 1, figsize=s.figure_size)
        p5 = ax.bar(index, SEC,  width, bottom=BC+BE+SPCc+SPCd, color=colors[4], label='SEC')
        p4 = ax.bar(index, SPCd, width, bottom=BC+BE+SPCc,      color=colors[3], label='SPCd')
        p3 = ax.bar(index, SPCc, width, bottom=BC+BE,           color=colors[2], label='SPCc')
        p2 = ax.bar(index, BE,   width, bottom=BC,              color=colors[1], label='BE')
        p1 = ax.bar(index, BC,   width,                         color=colors[0], label='BC')
        ax.set_ylabel(r'$€/MWh$')
        ax.xaxis.grid(False)
        plt.xticks(index + width/2., (
                                      'Without Storage\n100 %',
                                      'Without Storage\n 99.99 %',
                                      'Without Storage\n 99.9 %',
                                      'Without Storage\n99 %',
                                      'Unconstrained Storage',
                                      'Constrained Storage',
                                      ))
        labels = ('BC', 'BE', 'SPCc', 'SPCd', 'SEC')
        ax.legend(loc='best')
        s00 = r'$\alpha = {0:.2f} \quad \gamma = {1:.2f} \quad \beta^B = {2:.2f}$'.format(self.alpha, self.gamma, self.beta_b)
        s01 = (r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.{prec}f},'
                '\quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.{prec}f}$'.format(self.eta_in, self.eta_out, prec=2))
        s02 = (r'$\mathcal{{K}}^{{SPc}}={0:.{prec}f}, \mathcal{{K}}^{{SPd}}={1:.{prec}f}, '
                '\mathcal{{K}}^{{SE}}={2:.{prec}f}$'.format(self.K_SPc1, self.K_SPd1, self.K_SE1, prec=4))
        s03 = r'$E^{{NS}} = {0:.{prec}f}, \quad NS: {1:.0f}/{2:.0f}={3:.5f}\%$'.format(np.sum(self.E_NS),
                self.E_NS_hours, self.nhours, 100*self.E_NS_hours/self.nhours, prec=4)
        s0all = '\n'.join(('Constrained Storage Values:', s00, s01, s02, s03))
#         ax.text(5 + width/2, self.LCOE_B_1 + self.LCOE_S_1 + 1.0, s0all, ha='center', va='bottom',
#                 bbox=dict(facecolor='None', edgecolor='k'))
#         anchored_text0 = AnchoredText(s0all, loc=9, frameon=False)
#         ax.add_artist(anchored_text0)

        if show_numbers:
            def _add_text(ax, x, y, v):
                ax.text(x, y, '{0:.4f}'.format(v), va='center', ha='center')
            # Backup Capacity Text
            _add_text(ax, 0+width/2, BC[0]/2, self.LCOE_BC_2)
            _add_text(ax, 1+width/2, BC[1]/2, self.LCOE_BC_9999)
            _add_text(ax, 2+width/2, BC[2]/2, self.LCOE_BC_999)
            _add_text(ax, 3+width/2, BC[3]/2, self.LCOE_BC_99)
            _add_text(ax, 4+width/2, BC[4]/2, self.LCOE_BC_0)
            _add_text(ax, 5+width/2, BC[5]/2, self.LCOE_BC_1)
            # Backup Energy Text
            _add_text(ax, 0+width/2, BC[0] + BE[0]/2,self.LCOE_BE_2)
            _add_text(ax, 1+width/2, BC[1] + BE[1]/2,self.LCOE_BE_9999)
            _add_text(ax, 2+width/2, BC[2] + BE[2]/2,self.LCOE_BE_999)
            _add_text(ax, 3+width/2, BC[3] + BE[3]/2,self.LCOE_BE_99)
            _add_text(ax, 4+width/2, BC[4] + BE[4]/2,self.LCOE_BE_0)
            _add_text(ax, 5+width/2, BC[5] + BE[5]/2,self.LCOE_BE_1)
            # Storage Charge Power Text
            _add_text(ax, 4+width/2, BC[4] + BE[4] + SPCc[4]/2,self.LCOE_SPCc_0)
            _add_text(ax, 5+width/2, BC[5] + BE[5] + SPCc[5]/2,self.LCOE_SPCc_1)
            # Storage Discharge Power Text
            _add_text(ax, 4+width/2, BC[4] + BE[4] + SPCc[4] + SPCd[4]/2,self.LCOE_SPCd_0)
            _add_text(ax, 5+width/2, BC[5] + BE[5] + SPCc[5] + SPCd[5]/2,self.LCOE_SPCd_1),
            # Totals Text
            _add_text(ax, 0 + width/2, self.LCOE_B_2 + 0.1, self.LCOE_B_2)
            _add_text(ax, 1 + width/2, self.LCOE_B_9999 + 0.1, self.LCOE_B_9999)
            _add_text(ax, 2 + width/2, self.LCOE_B_999 + 0.1, self.LCOE_B_999)
            _add_text(ax, 3 + width/2, self.LCOE_B_99 + 0.1, self.LCOE_B_99)
            _add_text(ax, 4 + width/2, self.LCOE_B_0 + self.LCOE_S_0 + 0.1, self.LCOE_B_0 + self.LCOE_S_0),
            _add_text(ax, 5 + width/2, self.LCOE_B_1 + self.LCOE_S_1 + 0.1, self.LCOE_B_1 + self.LCOE_S_1),

        plt.show()
        fig.savefig(s.figures_folder + 'stacked_LCOE.pdf')
        plt.close()


    def capacity_test_KSE(self):
        colors = sns.color_palette()
        KSE_iter = np.linspace(7.0, 1.0, 61)
        self.E_NS_list = np.empty_like(KSE_iter)
        self.E_NS_hours_list = np.empty_like(KSE_iter)
        for i, KSE in enumerate(tqdm(KSE_iter)):
            self.set_storage_constraints(
                                        KSE=KSE,
                                        eta_in=p.prices_storage_bussar.EfficiencyCharge,
                                        eta_out=p.prices_storage_bussar.EfficiencyDischarge,
                                        add_curtailment=True,
                                        )
            self.calculate_storage()
            self.get_objectives()
            self.calculate_LCOE()
            self.calculate_E_NS()
            E_NS = np.sum(self.E_NS)
            E_NS_hours = self.E_NS_hours

            self.E_NS_list[i] = E_NS
            self.E_NS_hours_list[i] = E_NS_hours

        fig1, (ax1) = plt.subplots(1, 1, figsize=s.figure_size)
        fig2, (ax2) = plt.subplots(1, 1, figsize=s.figure_size)
        ax1.plot(KSE_iter, self.E_NS_list, label='$E^{NS, S}$')
        ax1.invert_xaxis()
        ax1.set_xlabel('$\mathcal{K}^{SE}$',fontsize=12)
        ax1.set_ylabel(r'$[\langle L_n \rangle]$')
        ax1.legend(loc='best',fontsize=12)
        fig1.savefig(s.figures_folder + 'E_NS_vs_KSE.pdf')
        ax2.invert_xaxis()
        ax2.set_xlabel('$\mathcal{K}^{SE}$',fontsize=12)
        ax2.set_ylabel(r'$\%$ of all hours')
        ax2.plot(KSE_iter, 100*self.E_NS_hours_list/self.nhours, label='$\mathrm{Hours}(E^{NS, S}>0)$')
        ax2.legend(loc='best', fontsize=12)
        fig2.savefig(s.figures_folder + 'E_NS_hours_vs_KSE.pdf')
        plt.show()

        return

    def capacity_test_KSPd(self):
        colors = sns.color_palette()
        KSPd_iter = np.linspace(0.5, 0, 1001)
        self.E_NS_list = np.empty_like(KSPd_iter)
        self.E_NS_hours_list = np.empty_like(KSPd_iter)
        for i, KSPd in enumerate(tqdm(KSPd_iter)):
            self.set_storage_constraints(KSPd=KSPd,
                                        eta_in=p.prices_storage_bussar.EfficiencyCharge,
                                        eta_out=p.prices_storage_bussar.EfficiencyDischarge,
                                        add_curtailment=True,
                                        )
            self.calculate_storage()
            self.get_objectives()
            self.calculate_LCOE()
            self.calculate_E_NS()
            E_NS = np.sum(self.E_NS)
            E_NS_hours = self.E_NS_hours

            self.E_NS_list[i] = E_NS
            self.E_NS_hours_list[i] = E_NS_hours

        fig1, (ax1) = plt.subplots(1, 1, figsize=s.figure_size)
        fig2, (ax2) = plt.subplots(1, 1, figsize=s.figure_size)
        ax1.plot(KSPd_iter, self.E_NS_list, label='$E^{NS, S}$')
        ax1.invert_xaxis()
        ax1.set_xlabel('$\mathcal{K}^{SPd}_{\max}$',fontsize=12)
        ax1.set_ylabel(r'$[\langle L_n \rangle]$')
        ax1.legend(loc='best',fontsize=12)
        fig1.savefig(s.figures_folder + 'E_NS_vs_KSPd.pdf')
        ax2.invert_xaxis()
        ax2.set_xlabel('$\mathcal{K}^{SPd}_{\max}$',fontsize=12)
        ax2.set_ylabel(r'$\%$ of all hours')
        ax2.semilogy(KSPd_iter, 100*self.E_NS_hours_list/self.nhours, label='$\mathrm{Hours}(E^{NS, S}>0)$')
#         ax2.plot(KSPd_iter, 100*self.E_NS_hours_list/self.nhours, label='$\mathrm{Hours}(E^{NS, S}>0)$')
        ax2.legend(loc='best', fontsize=12)
        fig2.savefig(s.figures_folder + 'E_NS_hours_vs_KSPd.pdf')
        plt.show()

        return


    def capacity_test_KSPc(self):
        colors = sns.color_palette()
        KSPc_iter = np.linspace(0.02, 0, 101)
        self.BE_list = np.empty_like(KSPc_iter)
        self.LCOE_list = np.empty_like(KSPc_iter)
        for i, KSPc in enumerate(tqdm(KSPc_iter)):
            self.set_storage_constraints(
                                        KSPc=KSPc,
                                        KSPd=0.17,
                                        eta_in=p.prices_storage_bussar.EfficiencyCharge,
                                        eta_out=p.prices_storage_bussar.EfficiencyDischarge,
                                        add_curtailment=True,
                                        )
            self.calculate_storage()
            self.get_objectives()
            self.calculate_LCOE()

            self.BE_list[i] = np.sum(self.Bs1)/8
            self.LCOE_list[i] = np.sum(self.LCOE1)

        fig1, (ax1) = plt.subplots(1, 1, figsize=s.figure_size)
        ax1.plot(KSPc_iter, self.BE_list, label='$E^B/y$')
        ax1.invert_xaxis()
        ax1.set_xlabel('$\mathcal{K}^{SPc}_{\max}$',fontsize=12)
        ax1.set_ylabel(r'$[\langle L_n \rangle]$')
        ax1.legend(loc='best',fontsize=12)
        ax1.set_xlim([KSPc_iter[0], KSPc_iter[-1]])
        fig1.savefig(s.figures_folder + 'BE_vs_KSPc.pdf')
        fig2, (ax2) = plt.subplots(1, 1, figsize=s.figure_size)
        ax2.plot(KSPc_iter, self.LCOE_list, label='$LCOE$')
        ax2.invert_xaxis()
        ax2.set_xlabel('$\mathcal{K}^{SPc}_{\max}$',fontsize=12)
        ax2.set_ylabel(r'$€$')
        ax2.legend(loc='best',fontsize=12)
        ax2.set_xlim([KSPc_iter[0], KSPc_iter[-1]])
        fig2.savefig(s.figures_folder + 'LCOE_vs_KSPc.pdf')
        plt.show()

        return


    def plot_no_storage_bars(self, show_numbers=True):
        colors = sns.color_palette()
        index = np.arange(4)
        width = 0.5
        # Sequence: 100, 99.99, 99.9, 99, constrained, unconstrained.
        # 2, 9999, 999, 99, 0, 1
        BC = np.array([
                       self.LCOE_BC_2,
                       self.LCOE_BC_9999,
                       self.LCOE_BC_999,
                       self.LCOE_BC_99,
                       ])
        BE = np.array([
                       self.LCOE_BE_2,
                       self.LCOE_BE_9999,
                       self.LCOE_BE_999,
                       self.LCOE_BE_99,
                       ])
        fig, (ax) = plt.subplots(1, 1, figsize=s.figure_size)
        p2 = ax.bar(index, BE,   width, bottom=BC,              color=colors[1], label='BE')
        p1 = ax.bar(index, BC,   width,                         color=colors[0], label='BC')
        ax.set_ylabel(r'$€/MWh$')
        ax.xaxis.grid(False)
        plt.xticks(index + width/2., (
                                      'Without Storage\n100 %',
                                      'Without Storage\n 99.99 %',
                                      'Without Storage\n 99.9 %',
                                      'Without Storage\n99 %',
                                      ))
        labels = ('BC', 'BE')
        ax.legend(loc='best')

        if show_numbers:
            def _add_text(ax, x, y, v):
                ax.text(x, y, '{0:.3f}'.format(v), va='center', ha='center')
            # Backup Capacity Text
            _add_text(ax, 0+width/2, BC[0]/2, self.LCOE_BC_2)
            _add_text(ax, 1+width/2, BC[1]/2, self.LCOE_BC_9999)
            _add_text(ax, 2+width/2, BC[2]/2, self.LCOE_BC_999)
            _add_text(ax, 3+width/2, BC[3]/2, self.LCOE_BC_99)
            # Backup Energy Text
            _add_text(ax, 0+width/2, BC[0] + BE[0]/2,self.LCOE_BE_2)
            _add_text(ax, 1+width/2, BC[1] + BE[1]/2,self.LCOE_BE_9999)
            _add_text(ax, 2+width/2, BC[2] + BE[2]/2,self.LCOE_BE_999)
            _add_text(ax, 3+width/2, BC[3] + BE[3]/2,self.LCOE_BE_99)
            # Storage Charge Power Text
            # Totals Text
            _add_text(ax, 0 + width/2, self.LCOE_B_2 + 0.1, self.LCOE_B_2)
            _add_text(ax, 1 + width/2, self.LCOE_B_9999 + 0.1, self.LCOE_B_9999)
            _add_text(ax, 2 + width/2, self.LCOE_B_999 + 0.1, self.LCOE_B_999)
            _add_text(ax, 3 + width/2, self.LCOE_B_99 + 0.1, self.LCOE_B_99)

        plt.show()
        fig.savefig(s.figures_folder + 'stacked_LCOE_no_storage.pdf')
        plt.close()

    def plot_final_timeseries(self):
        linew = 1
        fig, (ax) = plt.subplots(1, 1, figsize=s.figure_size)
        tt = range(len(self.S0))
        ax.plot(tt, self.S0, label='Unconstrained $S_n(t)$', color=self.colors[0])
        ax.plot(tt, self.S1, label='Constrained $S_n(t)$', color=self.colors[1])
#         ax.plot(tt, self.B, label=r'$G_n^B(t)$', color=self.colors[2])
#         ax.plot(tt, self.Bs0, label=r'Unconstrained $G_n^{{BS}}(t)$',
#                 color=self.colors[3])
#         ax.plot(tt, self.Bs1, label=r'Constrained $G_n^{{BS}}(t)$',
#                 color=self.colors[4])
#         ax.plot(tt, self.C, label=r'C', color=self.colors[4])
        ax.plot(tt, self.E_NS, 'k', label='Non-served storage energy $E^{NS,S}(t)$', color='k')

    # Strings for labels
        ax.legend(loc='best', fontsize=12)

        ax.set_xlabel(r'$t [y]$')
        ax.set_ylabel(r'$\left[\langle L_n \rangle \right]$')
        ax.set_xlim(self.tt[0], self.tt[-1])
        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[-1], 9))
        ax.set_xticklabels(range(0, 9))
        fig.savefig(s.figures_folder + 'storage_final.pdf')

        plt.show()
        plt.close('all')

L = LCOE_Objectives()
