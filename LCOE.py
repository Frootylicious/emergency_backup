############################################## TODO ################################################
####################################################################################################


import settings.settings as s
import settings.tools as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm
seaborn.set_style('whitegrid')

class LCOE_storage():
    '''
    Class to calculate the different costs associated with a storage.

    From a given solved network, the backup energy (BE), backup capacity (BC), storage power
    capacity (SPC) and storage energy capacity (SEC) are calculated. From there the LCOE is
    calculated for the whole system as a function of the backup capacity-beta.
    '''

    def __init__(self):
        # Variables --------------------------------------------------------------------------------
        self.eta_store = 0.75 # Efficiency of the hydrogen storage
        self.eta_dispatch = 0.58
        self.beta_capacity = 0.50 # Backup capacity in units of mean load.

        # Costs ------------------------------------------------------------------------------------
        self.DOLLAR_TO_EURO = 0.7532 # David
#         self.DOLLAR_TO_EURO = 0.9071 # Recent.

        # Loading network and setting timeseries ---------------------------------------------------
        # Loading the node object
        N = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        self.L_EU = N.f.load # Extracting the load of all nodes.
        self.all_load = np.sum(N.f.load) # Total load of all time.

        self.avg_L_EU = np.mean(np.sum(self.L_EU, axis=0)) # The average load of Europe.
        self.avg_L_n = np.mean(self.L_EU, axis=1) # The average load of each country.

        self.curtailment = N.f.curtailment
        # Curtailment in units of each country's average load.
        self.curtailment_relative = np.array([c / L for c, L in zip(self.curtailment, self.avg_L_n)])

        # Dividing each node's balancing by its average load.
        self.balancing = N.f.balancing
        self.balancing_relative = np.array([b / L for b, L in zip(self.balancing, self.avg_L_n)])

    def get_Sn(self):
        '''
        Calculate the storage size for each country and save it to a matrix.
        Also calculates the backup storage size with curtailment.

        self.Sn : all countries' storage filling level timeseries.
        self.S_all : a storage filling level timeseries for the combined Europe.
        '''
        self.Sn_relative = np.empty_like(self.balancing_relative)
        self.Bs_n_relative = np.empty_like(self.balancing_relative)
        for n, (b, c) in enumerate(zip(self.balancing_relative, self.curtailment_relative)):
            self.Sn_relative[n], self.Bs_n_relative[n] = t.storage_size_relative(b,
                    self.beta_capacity,
                    curtailment=c,
                    eta_in=self.eta_store,
                    eta_out=self.eta_dispatch)[1:]

            self.Sn = np.array([s * l for s, l in zip(self.Sn_relative, self.avg_L_n)])
        self.S_all = np.sum(self.Sn, axis=0)

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

    def get_SEC(self):
        '''
        Storage Energy (SE) [MWh] the summed storage for all nodes.
        '''
        self.SEC = -np.min(self.S_all)

    def get_SPC(self):
        '''
        Max charging and discharging rate.
        Max and min of diff of the filling level of the storage.
        '''
        (self.SPC_charge, self.SPC_discharge) = t.diff_max_min(self.S_all)

    def calculate_storage(self):
        self.get_Sn()
        self.get_BC()
        self.get_BE()
        self.get_SEC()
        self.get_SPC()

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
        # Storage Power Capacity (SPC) (fuel cells/electrolysis)
        SPC = np.max((np.abs(self.SPC_charge), np.abs(self.SPC_discharge)))
        # Storage capacity (SEC) (steel tanks)
        SEC = self.SEC


        # BACKUP ----------------------------------------------------------------------
        # Need Backup Energy in  MWh/year
        BE_per_year = self.BE / 8
        # Backup capacity in MW
        BC = self.BC
        # Costs:
        BE_costs = BE_per_year*asset_CCGT['OpExVariable'] * annualizationFactor(asset_CCGT['Lifetime'])
        BC_costs = self.BC*(asset_CCGT['CapExFixed'] * 1e6 + asset_CCGT['OpExFixed'] * 1e3 * annualizationFactor(asset_CCGT['Lifetime']))

        # Cost of electrolysis and fuel cells.
        SPC_costs = SPC * (737 + annualizationFactor(20) * 12.2) * 1e3 * self.DOLLAR_TO_EURO
        # Cost of steel tanks.
        SEC_costs = SEC * 11.2 * 1e3 * self.DOLLAR_TO_EURO

        scalingFactor_30 = self.all_load / 8 *annualizationFactor(30)
        scalingFactor_20 = self.all_load / 8 *annualizationFactor(20)

        self.LCOE_BE = BE_costs / scalingFactor_30
        self.LCOE_BC = BC_costs / scalingFactor_30

        self.LCOE_SPC = SPC_costs / scalingFactor_20
        self.LCOE_SEC = SEC_costs / scalingFactor_20

        self.TOTAL_LCOE_B = self.LCOE_BE + self.LCOE_BC
        self.TOTAL_LCOE_S = self.LCOE_SPC + self.LCOE_SEC

        self.TOTAL_LCOE = self.TOTAL_LCOE_B + self.TOTAL_LCOE_S

    def test(self, beta_capacity=0.50):
        self.beta_capacity=beta_capacity
        self.calculate_storage()
        print('----------- Beta is: {:.4f} -----------'.format(self.beta_capacity))
        return


    def test_all(self, beta_list=np.linspace(0.00, 1.50, 3001), save=True):
        self.beta_list = beta_list
        n = len(beta_list)
        self.BC_list = np.empty(n)
        self.BE_list = np.empty(n)
        self.SEC_list = np.empty(n)
        self.SPC_list = np.empty(n)

        self.LCOE_BC_list = np.empty(n)
        self.LCOE_BE_list = np.empty(n)
        self.LCOE_SPC_list = np.empty(n)
        self.LCOE_SEC_list = np.empty(n)

        for i, b in tqdm(enumerate(beta_list)):
            self.test(beta_capacity=b)
            self.calculate_costs()

            self.BC_list[i] = self.BC
            self.BE_list[i] = self.BE
            self.SEC_list[i] = self.SEC
            self.SPC_list[i] = np.max((np.abs(self.SPC_charge), np.abs(self.SPC_discharge)))

            self.LCOE_BE_list[i] = self.LCOE_BE
            self.LCOE_BC_list[i] = self.LCOE_BC
            self.LCOE_SPC_list[i] = self.LCOE_SPC
            self.LCOE_SEC_list[i] = self.LCOE_SEC

        if save:
            np.save(s.results_folder + 'LCOE/' + 'BC_list', self.BC_list)
            np.save(s.results_folder + 'LCOE/' + 'BE_list', self.BE_list)
            np.save(s.results_folder + 'LCOE/' + 'SEC_list', self.SEC_list)
            np.save(s.results_folder + 'LCOE/' + 'SPC_list', self.SPC_list)

            np.save(s.results_folder + 'LCOE/' + 'LCOE_BE_list', self.LCOE_BE_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_BC_list', self.LCOE_BC_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_SPC_list', self.LCOE_SPC_list)
            np.save(s.results_folder + 'LCOE/' + 'LCOE_SEC_list', self.LCOE_SEC_list)

# --------------------------------------------------------------------------------------------------

L = LCOE_storage()
# L.test_all(save=True)
# --------------------------------------------------------------------------------------------------

def plot_timeseries():
    L.get_Sn()
    L.get_BE()
    plt.plot(L.balancing_relative[18], 'r')
    plt.plot(L.Bs_n_relative[18], 'b')
    plt.plot(L.Sn_relative[18], 'g')
    plt.plot(L.curtailment_relative[18], 'c')
    plt.show()


def plot_results():
    beta_list = np.linspace(0, 1.5, 3001)
    path = s.results_folder + 'LCOE/'

    BC = np.load(path + 'BC_list.npy')
    BE = np.load(path + 'BE_list.npy')
    SEC = np.load(path + 'SEC_list.npy')
    SPC = np.load(path + 'SPC_list.npy')

    LCOE_BC = np.load(path + 'LCOE_BC_list.npy')
    LCOE_BE = np.load(path + 'LCOE_BE_list.npy')
    LCOE_SEC = np.load(path + 'LCOE_SEC_list.npy')
    LCOE_SPC = np.load(path + 'LCOE_SPC_list.npy')

    beta_list = np.linspace(0, 1.5, len(BC))

    backup_costs = LCOE_BC + LCOE_BE
    storage_costs = LCOE_SPC + LCOE_SEC
    total_costs = backup_costs + storage_costs

    # Plotting stacked plot of all LCOE
    fig1, (ax1) = plt.subplots(1, 1, sharex=True)
    l = ('LCOE BC', 'LCOE BE', 'LCOE SPC', 'LCOE SEC')
    ax1.stackplot(beta_list, LCOE_BC, LCOE_BE, LCOE_SPC, LCOE_SEC, labels=l)
    ax1.set_xlim([0, 1.5])
    ax1.set_ylim([0, 30])
    ax1.set_ylabel(r'$€/MWh$')
    ax1.set_xlabel(r'$\beta^B$')
    ax1.legend()
    fig1.savefig(path + 'stacked_LCOE.pdf', bbox_inches='tight')

    fig2, (ax21, ax22, ax23, ax24) = plt.subplots(4, 1, sharex=True)
    ax21.plot(beta_list, BC, label = 'Backup Capacity')
    ax22.plot(beta_list, BE, label = 'Backup Energy')
    ax23.plot(beta_list, SPC, label = 'Storage Power')
    ax24.plot(beta_list, SEC, label = 'Storage Capaticy')
    ax34.set_xlabel(r'$\beta^B$')
    ax22.set_xlim([0, 1.5])
    ax21.legend()
    ax22.legend()
    ax23.legend()
    ax24.legend()
    fig2.savefig(path + 'BC_BE_SPC_SEC.pdf', bbox_inches='tight')

    fig3, (ax31, ax32, ax33, ax34) = plt.subplots(4, 1, sharex=True, sharey=True)
    ax31.plot(beta_list, LCOE_BC, label = 'LCOE Backup Capacity')
    ax32.plot(beta_list, LCOE_BE, label = 'LCOE Backup Energy')
    ax33.plot(beta_list, LCOE_SPC, label = 'LCOE Storage Power')
    ax34.plot(beta_list, LCOE_SEC, label = 'LCOE Storage Capaticy')
    ax34.set_xlabel(r'$\beta^B$')
    fig3.text(0.04, 0.5, '$€/MWh$', va='center', rotation='vertical')
    ax34.set_ylim([0, 30])
    ax34.set_xlim([0, 1.5])
    ax31.legend()
    ax32.legend()
    ax33.legend()
    ax34.legend()
    fig3.savefig(path + 'LCOE_BC_BE_SPC_SEC.pdf', bbox_inches='tight')

    plt.show()

# plot_results()
