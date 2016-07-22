############################################## TODO ################################################
# Lad så at hvis beta_capacity skifter, så skal den genregne resultaterne. Måske er det fint at lade
# det være som det er.
#
# Regn den endelige pris ud.
####################################################################################################

import settings.settings as s
import settings.tools as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LCOE_storage():
    '''
    Class to calculate the different costs associated with a storage.
    '''

    def __init__(self):
        # Variables --------------------------------------------------------------------------------
        self.eta_store = 0.75
        self.eta_dispatch = 0.58
        self.beta_capacity = 0.50

        # Costs ------------------------------------------------------------------------------------
        self.DOLLAR_TO_EURO = 0.7532 # David
#         self.DOLLAR_TO_EURO = 0.9071

        # Loading network and setting timeseries ---------------------------------------------------
        # Loading the node object
        N = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        # Extracting the load of all nodes.
        self.L_EU = N.f.load
        # Total load of all time.
        self.all_load = np.sum(N.f.load)
        # The average load of Europe.
        self.avg_L_EU = np.mean(np.sum(self.L_EU, axis=0))
        # The average load of each country.
        self.avg_L_n = np.mean(self.L_EU, axis=1)

        self.curtailment = N.f.curtailment
        # Curtailment in units of each country's average load.
        self.curtailment_relative = np.array([c / L for c, L in zip(self.curtailment, self.avg_L_n)])

        # Dividing each node's balancing by its average load.
        self.balancing = N.f.balancing
        self.balancing_relative = np.array([b / L for b, L in zip(self.balancing, self.avg_L_n)])

    def calculate_storage(self):
        self.get_S_n()
        self.get_BC()
        self.get_BE()
        self.get_K_ES()
        self.get_K_PS()

    def get_S_n(self):
        '''
        Calculate the storage size for each country and save it to a matrix.
        Also calculates the backup storage size with curtailment.

        self.S_n : all countries' storage filling level timeseries.
        self.S_all : a storage filling level timeseries for the combined Europe.
        '''
        self.S_n_relative = np.empty_like(self.balancing_relative)
        self.Bs_n_relative = np.empty_like(self.balancing_relative)
        for n, (b, c) in enumerate(zip(self.balancing_relative, self.curtailment_relative)):
            self.S_n_relative[n], self.Bs_n_relative[n] = t.storage_size_relative(b,
                                                                        self.beta_capacity,
                                                                        curtailment=c,
                                                                        eta_in=self.eta_store,
                                                                        eta_out=self.eta_dispatch)[1:]

        self.S_n = np.array([s * l for s, l in zip(self.S_n_relative, self.avg_L_n)])
        self.S_all = np.sum(self.S_n, axis=0)

    def get_BC(self):
        '''
        Backup Capacity (BC)  [MW] which is \beta * <L_n> which is summed for each country
        '''
        self.BC = np.sum([self.beta_capacity * L for L in self.avg_L_n])

    def get_BE(self):
        '''
        Calculating the backup energy BE.

        The beta_capacity decides the capacity for each country.
        To calculate the real backup energy with a storage, we need to account for storage charging
        and the capacity of the backup.

        The curtailment is used for charging, if the storage is not full and there is curtailment.

        If curtailment, if storage - use curtailment.
        If no curtailment, is storage - use backup.
        '''
        self.BE = np.sum([b * l for b, l in zip(self.Bs_n_relative, self.avg_L_n)])

    def get_K_ES(self):
        '''
        Storage Energy (SE) [MWh] the summed storage for all nodes.
        '''
        self.K_ES = -np.min(self.S_all)

    def get_K_PS(self):
        '''
        Max charging and discharging rate.
        Max and min of diff of the filling level of the storage.
        '''
        (self.K_PS_charge, self.K_PS_discharge) = t.diff_max_min(self.S_all)


    def calculate_costs(self):
        ## Cost assumptions: // Source: Rolando PHD thesis, table 4.1, page 109. Emil's thesis.
        asset_CCGT = {
            'Name': 'CCGT backup',
            'CapExFixed': 0.9, #Euros/W
            'OpExFixed': 4.5, #Euros/kW/year
            'OpExVariable': 56.0, #Euros/MWh/year
            'Lifetime': 30 #years
            }


        def annualizationFactor(lifetime, r=4.0):
            """Lifetime in years and r = rate in percent."""
            if r==0: return lifetime
            return (1-(1+(r/100.0))**-lifetime)/(r/100.0)

        # STORAGE --------------------------------------------------------------------------
        # Storage Power (fuel cells/electrolysis)
        SP = np.max((np.abs(self.K_PS_charge), np.abs(self.K_PS_discharge)))
        # Storage capacity (steel tanks)
        SC = self.K_ES


        # BACKUP ----------------------------------------------------------------------
        # Need Backup Energy in  MWh/year
        BE_per_year = self.BE / 8
        # Backup capacity in MW
        BC = self.BC
        # Costs:
        BE_costs = BE_per_year*asset_CCGT['OpExVariable'] * annualizationFactor(asset_CCGT['Lifetime'])
        BC_costs = self.BC*(asset_CCGT['CapExFixed'] * 1e6 + asset_CCGT['OpExFixed'] * 1e3 * annualizationFactor(asset_CCGT['Lifetime']))

        # Cost of electrolysis and fuel cells.
        SP_costs = SP * (737 + annualizationFactor(20) * 12.2) * 1e3 * self.DOLLAR_TO_EURO
        # Cost of steel tanks.
        SC_costs = SC * 11.2 * 1e3 * self.DOLLAR_TO_EURO

        scalingFactor_30 = self.all_load / 8 *annualizationFactor(30)
        scalingFactor_20 = self.all_load / 8 *annualizationFactor(20)

        self.LCOE_BE = BE_costs / scalingFactor_30
        self.LCOE_BC = BC_costs / scalingFactor_30

        self.LCOE_SP = SP_costs / scalingFactor_20
        self.LCOE_SC = SC_costs / scalingFactor_20

        self.TOTAL_LCOE_B = self.LCOE_BE + self.LCOE_BC
        self.TOTAL_LCOE_S = self.LCOE_SP + self.LCOE_SC

        self.TOTAL_LCOE = self.TOTAL_LCOE_B + self.TOTAL_LCOE_S


    def test(self, beta_capacity=0.50):
        self.beta_capacity=beta_capacity
        self.calculate_storage()
        print('----------- Beta is: {0} -----------'.format(self.beta_capacity))
        return


    def test_all(self, beta_list=np.linspace(0.00, 1.50, 1501), save=True):
        self.beta_list = beta_list
        n = len(beta_list)
        self.BC_list = np.empty(n)
        self.BE_list = np.empty(n)
        self.SC_list = np.empty(n)
        self.SP_list = np.empty(n)

        self.LCOE_BC_list = np.empty(n)
        self.LCOE_BE_list = np.empty(n)
        self.LCOE_SP_list = np.empty(n)
        self.LCOE_SC_list = np.empty(n)

        for i, b in tqdm(enumerate(beta_list)):
            self.test(beta_capacity=b)

            self.BC_list[i] = self.BC
            self.BE_list[i] = self.BE
            self.SC_list[i] = self.K_ES
            self.SP_list[i] = np.max((np.abs(self.K_PS_charge), np.abs(self.K_PS_discharge)))

            self.calculate_costs()
            self.LCOE_BE_list[i] = self.LCOE_BE
            self.LCOE_BC_list[i] = self.LCOE_BC
            self.LCOE_SP_list[i] = self.LCOE_SP
            self.LCOE_SC_list[i] = self.LCOE_SC

        if save:
            np.save(s.results_folder + 'LCOE/' + 'BC_list', self.BC_list)
            np.save(s.results_folder + 'LCOE/' + 'BE_list', self.BE_list)
            np.save(s.results_folder + 'LCOE/' + 'SC_list', self.SC_list)
            np.save(s.results_folder + 'LCOE/' + 'SP_list', self.SP_list)

            np.save(s.results_folder + 'LCOE/' + 'LCOE_BE_list', self.LCOE_BE_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_BC_list', self.LCOE_BC_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_SP_list', self.LCOE_SP_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_SC_list', self.LCOE_SC_list)

# --------------------------------------------------------------------------------------------------

L = LCOE_storage()
L.test_all(save=True)
# --------------------------------------------------------------------------------------------------

def plotthings():
        
    L.get_S_n()
    L.get_BE()
    plt.plot(L.balancing_relative[18], 'r')
    plt.plot(L.Bs_n_relative[18], 'b')
    plt.plot(L.S_n_relative[18], 'g')
    plt.plot(L.curtailment_relative[18], 'c')
    plt.show()


def plot_results():
    beta_list = np.linspace(0, 1.5, 151)
    path = s.results_folder + 'LCOE/'

    BC = np.load(path + 'BC_list.npy')
    BE = np.load(path + 'BE_list.npy')
    SC = np.load(path + 'SC_list.npy')
    SP = np.load(path + 'SP_list.npy')

    LCOE_BC = np.load(path + 'LCOE_BC_list.npy')
    LCOE_BE = np.load(path + 'LCOE_BE_list.npy')
    LCOE_SC = np.load(path + 'LCOE_SC_list.npy')
    LCOE_SP = np.load(path + 'LCOE_SP_list.npy')

    beta_list = np.linspace(0, 1.5, len(BC))

    backup_costs = LCOE_BC + LCOE_BE
    storage_costs = LCOE_SP + LCOE_SC
    total_costs = backup_costs + storage_costs

    # Plotting stacked plot of all LCOE
    fig1, (ax1) = plt.subplots(1, 1, sharex=True)
    l = ('LCOE BC', 'LCOE BE', 'LCOE SP', 'LCOE SC')
    ax1.stackplot(beta_list, LCOE_BC, LCOE_BE, LCOE_SP, LCOE_SC, labels=l)
    ax1.set_xlim([0, 1.5])
    ax1.set_ylim([0, 30])
    ax1.set_ylabel(r'$€/MWh$')
    ax1.set_xlabel(r'$\beta^B$')
    ax1.legend()
    fig1.savefig(path + 'stacked_LCOE.pdf')

    fig2, (ax21, ax22, ax23, ax24) = plt.subplots(4, 1, sharex=True)
    ax21.plot(beta_list, BC, label = 'Backup Capacity')
    ax22.plot(beta_list, BE, label = 'Backup Energy')
    ax23.plot(beta_list, SP, label = 'Storage Power')
    ax24.plot(beta_list, SC, label = 'Storage Capaticy')
    ax22.set_xlim([0, 1.5])
    ax21.legend()
    ax22.legend()
    ax23.legend()
    ax24.legend()
    fig2.savefig(path + 'BC_BE_SP_SC.pdf')

    fig3, (ax31, ax32, ax33, ax34) = plt.subplots(4, 1, sharex=True, sharey=True)
    ax31.plot(beta_list, LCOE_BC, label = 'LCOE Backup Capacity')
    ax32.plot(beta_list, LCOE_BE, label = 'LCOE Backup Energy')
    ax33.plot(beta_list, LCOE_SP, label = 'LCOE Storage Power')
    ax34.plot(beta_list, LCOE_SC, label = 'LCOE Storage Capaticy')
    ax34.set_xlabel(r'$\beta^B$')
    fig3.text(0.04, 0.5, '$€/MWh$', va='center', rotation='vertical')
    ax34.set_ylim([0, 30])
    ax34.set_xlim([0, 1.5])
    ax31.legend()
    ax32.legend()
    ax33.legend()
    ax34.legend()
    fig3.savefig(path + 'LCOE_BC_BE_SP_SC.pdf')

    plt.show()

# plot_results()
