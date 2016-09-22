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
    def __init__(self):
        # Decimals.
        self.decimals_variables = 1
        self.decimals_objectives = 2
        self.decimals_LCOE = 3

        # Do stuff.
        self.do_all()

    def do_all(self):
        self.set_all()
        self.calculate_all()
        self.print_all()
        self.plot_all()

    def set_all(self):
        self.set_network_parameters()
        self.set_storage_constraints()

    def calculate_all(self):
        self.load_network()
        self.calculate_storage()
        self.get_objectives()
        self.calculate_LCOE()

    def print_all(self):
        self.print_objectives()
        self.print_costs()

    def plot_all(self):
        self.plot_timeseries()
        self.plot_bar()

    # Setting parameters for network.
    def set_network_parameters(self, a=0.80, g=1.00, b=0.70):
        self.alpha = a
        self.gamma = g
        self.beta_b = b

    # Setting constraints for storage.
    def set_storage_constraints(self,
                                KSPc=0.06,
                                KSPd=0.13,
                                eta_in=p.prices_storage_bussar.EfficiencyCharge,
                                eta_out=p.prices_storage_bussar.EfficiencyDischarge,
                                KSE=np.inf):
        self.KSPc = KSPc
        self.KSPd = KSPd
        self.eta_in = eta_in
        self.eta_out = eta_out
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
        self.C = self.N.f.curtailment[18]
        self.C /= self.avg_L_DE
        self.nhours = self.N.f.nhours[18]
        self.tt = range(len(self.B))

    # Calculate storage and objectives -------------------------------------------------------------
    def calculate_storage(self):
        (self.S_max0, self.S0, self.Bs0) = t.storage_size(self.B, self.beta_b)
        # Constrained
        (self.S_max1, self.S1,self. Bs1) = t.storage_size(self.B, self.beta_b,
                                           curtailment=self.C,
                                           eta_in=self.eta_in,
                                           eta_out=self.eta_out,
                                           charging_capacity=self.KSPc*self.eta_in,
                                           discharging_capacity=self.KSPd/self.eta_out,
                                           energy_capacity=self.KSE,
                                           )

        self.E_NS = -(self.S0 - self.S1).clip(max=0)
        self.E_NS[np.abs(self.E_NS) < 1e-10] = 0
        self.E_NS_hours = np.count_nonzero(self.E_NS)

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
        # Without storage
        self.LCOE2 = t.calculate_costs(np.max(self.B), np.sum(self.B), 1, 1, 1, self.L_DE,
                                  p.prices_backup_leon, p.prices_storage_bussar)
        (self.LCOE_BC_2, self.LCOE_BE_2) = self.LCOE2[:2]
        self.LCOE_B_2 = np.sum((self.LCOE_BC_2 + self.LCOE_BE_2))

    # Printing results -----------------------------------------------------------------------------
    def print_objectives(self):
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
        s8 = 'NO storage: ------------------'
        s9 = _get_objective_string(np.max(self.B), np.sum(self.B), 0, 0, 0, 0, 0)
        s10 = '\nE^NS    = {0:.2f}\nhours: {1}/{2} \n        = {3:.4f}'.format(np.sum(self.E_NS),
                                                                                      self.E_NS_hours,
                                                                                      self.nhours,
                                                                                      self.E_NS_hours/self.nhours)
        s = '\n'.join(('\n', s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10))
        print(s)

    def print_costs(self):
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
        s6 = 'NO storage: ------------------'
        s7 = _get_costs_string(self.LCOE_B_2, 0,
                               self.LCOE_BC_2, self.LCOE_BE_2,
                               0, 0, 0)
        s = '\n'.join(('\n', s0, s1, s2, s3, s4, s5, s6, s7))
        print(s)


    # Plotting ------ ------------------------------------------------------------------------------
    def plot_timeseries(self):
        linew = 1
        fig0, (ax0) = plt.subplots(1, 1)
        ax0.plot(range(len(self.S0)), self.S0)
        ax0.plot(range(len(self.S1)), self.S1)
        ax0.plot(range(len(self.B)), self.B, label=r'$G_n^B$')
        ax0.plot(range(len(self.Bs0)), self.Bs0, label=r'Unconstrained $G_n^{{BS}}$')
        ax0.plot(range(len(self.Bs1)), self.Bs1, label=r'Constrained $G_n^{{BS}}$')
        ax0.set_xlim(self.tt[0], self.tt[-1])
    # Strings for labels
        s01 = r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.2f}, \quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.2f}$'
        s02 = r'$C = {2}$'
        s03 = r'$\mathcal{{K}}^{{SPc}}={3}, \mathcal{{K}}^{{SPd}}={4}, \mathcal{{K}}^{{SE}}={5}$'
        s0all = '\n'.join((s01, s02, s03))
    # Making the string pretty with infty in latex
        ax0.lines[0].set_label(s0all.format(1, 1, 0, '\infty', '\infty', '\infty'))
        ax0.lines[1].set_label(s0all.format(self.eta_in, self.eta_out, 'C', self.KSPc if self.KSPc != np.inf else '\infty',
                                                                self.KSPd if self.KSPd != np.inf else '\infty',
                                                                self.KSE if self.KSE != np.inf else '\infty'))
        ax0.legend(loc='lower center', fontsize=12)
    # Strings for objectives
        s11 = r'$\beta^B = {0}, \quad \alpha = {1}, \quad \gamma = {2}$'.format(self.beta_b, self.alpha, self.gamma)
        s12 = '\nUnconstrained: '
        s13 = r'$\mathcal{{K}}^{{SPc}}={0:.2f},\quad \mathcal{{K}}^{{SPd}}={1:.2f}$'
        s17 = r'$\mathcal{{K}}^{{SE}} = {0:.2f}$'
        s14 = '\nConstrained: '
        s15 = r'$E^{{NS}} = {0:.2f}$'.format(np.sum(self.E_NS))
        s181 = '\n Differences: '
        s182 = r'$\Delta \mathcal{{K}}^{{SPc}} = {0:.2f}$'.format(self.K_SPc1 - self.K_SPc0)
        s183 = r'$\Delta \mathcal{{K}}^{{SPd}} = {0:.2f}$'.format(self.K_SPd1 - self.K_SPd0)
        s184 = r'$\Delta \mathcal{{K}}^{{SE}} = {0:.2f}$'.format(self.K_SE1 - self.K_SE0)
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
        ax0.add_artist(anchored_text1)

        ax0.set_xlabel(r'$t [y]$')
        ax0.set_ylabel(r'$S(t)\left[\langle L_n \rangle \right]$')
        ax0.set_xlim(self.tt[0], self.tt[-1])
        ax0.set_xticks(np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[-1], 9))
        ax0.set_xticklabels(range(0, 9))
        fig0.savefig(s.figures_folder + 'storage.pdf')

        plt.show()
        plt.close('all')

    def plot_bar(self):
        colors = sns.color_palette()
        index = np.arange(3)
        width = 0.5
        BC = np.array([self.LCOE_BC_0, self.LCOE_BC_1, self.LCOE_BC_2])
        BE = np.array([self.LCOE_BE_0, self.LCOE_BE_1, self.LCOE_BE_2])
        SPCc = np.array([self.LCOE_SPCc_0, self.LCOE_SPCc_1, 0])
        SPCd = np.array([self.LCOE_SPCd_0, self.LCOE_SPCd_1, 0])
        SEC = np.array([self.LCOE_SEC_0, self.LCOE_SEC_1, 0])
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
        ax1.legend((p5[0], p4[0], p3[0], p2[0], p1[0]), labels, loc='best')
        s01 = (r'$\widetilde{{\eta}}^\mathrm{{in}} = {0:.{prec}f},'
                '\quad \widetilde{{\eta}}^\mathrm{{out}} = {1:.{prec}f}$'.format(self.eta_in, self.eta_out, prec=2))
        s02 = (r'$\mathcal{{K}}^{{SPc}}={0:.{prec}f}, \mathcal{{K}}^{{SPd}}={1:.{prec}f}, '
                '\mathcal{{K}}^{{SE}}={2:.{prec}f}$'.format(self.K_SPc1, self.K_SPd1, self.K_SE1, prec=4))
        s03 = r'$E^{{NS}} = {0:.{prec}f}, \quad NS: {1:.0f}/{2:.0f}={3:.{prec}f}$'.format(np.sum(self.E_NS),
                self.E_NS_hours, self.nhours, self.E_NS_hours/self.nhours, prec=4)
        s0all = '\n'.join((s01, s02, s03))
        anchored_text0 = AnchoredText(s0all, loc=9, frameon=False)
        ax1.add_artist(anchored_text0)
        plt.show()
        fig1.savefig(s.figures_folder + 'stacked_LCOE.pdf')
        plt.close()

L = LCOE_Objectives()
