############################################## TODO ################################################
# Something wrong with Storage charge power.
####################################################################################################


import settings.settings as s
import settings.tools as t
import settings.prices as p
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
        self.beta_capacity = 1.00 # Backup capacity in units of mean load.

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

        # Dividing each node's backup by its average load.
        self.backup = N.f.balancing
        self.backup_relative = np.array([b / L for b, L in zip(self.backup, self.avg_L_n)])

    def get_Sn(self, beta_capacity):
        '''
        Calculate the storage size for each country and save it to a matrix.
        Also calculates the backup storage size with curtailment.

        self.Sn : all countries' storage filling level timeseries.
        self.S_all : a storage filling level timeseries for the combined Europe.
        '''
        self.Sn_relative = np.empty_like(self.backup_relative)
        self.Bs_n_relative = np.empty_like(self.backup_relative)
        for n, (b, c) in enumerate(zip(self.backup_relative, self.curtailment_relative)):
            self.Sn_relative[n], self.Bs_n_relative[n] = t.storage_size_relative(b,
                    beta_capacity,
                    curtailment=c,
                    eta_in=self.eta_store,
                    eta_out=self.eta_dispatch)[1:]

            self.Sn = np.array([s * l for s, l in zip(self.Sn_relative, self.avg_L_n)])
        self.S_all = np.sum(self.Sn, axis=0)

    def get_BC(self, beta_capacity):
        '''
        Backup Capacity (BC)  [MW] which is \beta * <L_n> which is summed for each country
        '''
        self.BC = np.sum([beta_capacity * L for L in self.avg_L_n])

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

    def get_SPC(self):
        '''
        Max charging and discharging rate.
        Max and min of diff of the filling level of the storage.
        '''
        (self.SPC_c, self.SPC_d) = t.diff_max_min(self.S_all)

    def get_SEC(self):
        '''
        Storage Energy (SE) [MWh] the summed storage for all nodes.
        '''
        self.SEC = -np.min(self.S_all)


    def calculate_storage(self, beta_capacity):
        self.get_Sn(beta_capacity)
        self.get_BC(beta_capacity)
        self.get_BE()
        self.get_SEC()
        self.get_SPC()

    def calculate_costs(self, BC, BE, SPC_c, SPC_d, SEC, prices_backup, prices_storage):
        ## Cost assumptions: // Source: Rolando PHD thesis, table 4.1, page 109. Emil's thesis.

        def _annualizationFactor(lifetime, r=4.0):
            """Lifetime in years and r = rate in percent."""
            if r==0: return lifetime
            return (1-(1+(r/100.0))**-lifetime)/(r/100.0)


        # BACKUP ----------------------------------------------------------------------
        # Need Backup Energy in  MWh/year
        BE_per_year = BE / 8
        # Backup capacity in MW
        # Costs:
        BE_costs = BE_per_year * prices_backup.OpExVariable * _annualizationFactor(prices_backup.Lifetime)
#         BE_costs = BE_per_year*prices_backup['OpExVariable'] * _annualizationFactor(prices_backup['Lifetime'])
        BC_costs = BC * (prices_backup.CapExFixed + prices_backup.OpExFixed * _annualizationFactor(prices_backup.Lifetime))
#         BC_costs = BC*(prices_backup['CapExFixed'] * 1e6 + prices_backup['OpExFixed'] * 1e3 * _annualizationFactor(prices_backup['Lifetime']))

        # Cost of electrolysis and fuel cells.
#         if prices_storage['ChargeDischargeDifferent'] == False:
#             SPC_costs = SPC * (prices_storage['Cap_Power'] + _annualizationFactor(20) * prices_storage['O&M_Power']) * 1e3
#         else:
#             SPC_costs_charge = np.abs(SPC_c) * (prices_storage['Cap_Power_Charge'] +
#                     _annualizationFactor(prices_storage['Lifetime']) * prices_storage['O&M_Power']) * 1e3
#             SPC_costs_discharge = np.abs(SPC_d) * (prices_storage['Cap_Power_Discharge'] +
#                     _annualizationFactor(prices_storage['Lifetime']) * prices_storage['O&M_Power']) * 1e3
#             SPC_costs = SPC_costs_charge + SPC_costs_discharge

        # STORAGE --------------------------------------------------------------------------
        # Storage Power Capacity - charge
        SPC_c_costs = np.abs(SPC_c) * (prices_storage.CapPowerCharge +
                                       _annualizationFactor(prices_storage.LifetimeCharge) *
                                       prices_storage.OMPower) * 1e3
        # Storage Power Capacity - discharge
        SPC_d_costs = np.abs(SPC_d) * (prices_storage.CapPowerDischarge +
                                       _annualizationFactor(prices_storage.LifetimeDischarge) *
                                       prices_storage.OMPower) * 1e3

        # Total costs for the Power Capacity = Charge Capacity costs + Discharge Capacity costs
        SPC_costs = SPC_c_costs + SPC_d_costs

        # Cost of steel tanks for storing hydrogen.
        SEC_costs = SEC * prices_storage.CapStorage * 1e3

        # Scaling factors
        sf_backup = self.all_load / 8 * _annualizationFactor(prices_backup.Lifetime)
        sf_storage_charge = self.all_load / 8 * _annualizationFactor(prices_storage.LifetimeCharge)
        sf_storage_discharge = self.all_load / 8 * _annualizationFactor(prices_storage.LifetimeDischarge)
        sf_storage_storage = self.all_load / 8 * _annualizationFactor(prices_storage.LifetimeStorage)

#         LCOE_BC = BC_costs / scalingFactor_backup
#         LCOE_BE = BE_costs / scalingFactor_backup

        LCOE_BC = BC_costs / sf_backup if sf_backup != 0 else 0
        LCOE_BE = BE_costs / sf_backup if sf_backup != 0 else 0
        LCOE_SPC_c = SPC_c_costs / sf_storage_charge if sf_storage_charge != 0 else 0
        LCOE_SPC_d = SPC_d_costs / sf_storage_discharge if sf_storage_discharge != 0 else 0
        LCOE_SEC = SEC_costs / sf_storage_storage if sf_storage_storage != 0 else 0


        return(LCOE_BC, LCOE_BE, LCOE_SPC_c, LCOE_SPC_d, LCOE_SEC)

    def test(self, beta_capacity=0.50):
        self.beta_capacity=beta_capacity
        self.calculate_storage(beta_capacity=beta_capacity)
        print('\n----------- Beta is: {:.4f} -----------'.format(self.beta_capacity))
        return


    def test_all(self, beta_list=np.linspace(0.00, 1.50, 31), solve_lists=False, save_lists=False,
            solve_LCOE=True, save_LCOE=False):
        prices_backup = p.prices_backup_leon
        prices_storage = p.prices_storage_david
        n = len(beta_list)
        if solve_lists:
            print('Solving BC, BE, SPC_c, SPC_d and SEC...')
            print('{0} iterations in total...'.format(n))
            self.beta_list = beta_list
            self.BC_list = np.empty(n)
            self.BE_list = np.empty(n)
            self.SPC_c_list = np.empty(n)
            self.SPC_d_list = np.empty(n)
            self.SEC_list = np.empty(n)

            for i, b in tqdm(enumerate(beta_list)):
                self.test(beta_capacity=b)

                BC = self.BC
                BE = self.BE
                SPC_c = np.abs(self.SPC_c)
                SPC_d = np.abs(self.SPC_d)
                SEC = self.SEC

                self.BC_list[i] = BC
                self.BE_list[i] = BE
                self.SPC_c_list[i] = SPC_c
                self.SPC_d_list[i] = SPC_d
                self.SEC_list[i] = SEC
            
            if save_lists:
                np.save(s.results_folder + 'LCOE/' + 'BC_list_{0}'.format(n), self.BC_list)
                np.save(s.results_folder + 'LCOE/' + 'BE_list_{0}'.format(n), self.BE_list)
                np.save(s.results_folder + 'LCOE/' + 'SEC_list_{0}'.format(n), self.SEC_list)
                np.save(s.results_folder + 'LCOE/' + 'SPC_c_list_{0}'.format(n), self.SPC_c_list)
                np.save(s.results_folder + 'LCOE/' + 'SPC_d_list_{0}'.format(n), self.SPC_d_list)

        else:
            self.BC_list = np.load(s.results_folder + 'LCOE/' + 'BC_list_{0}.npy'.format(n))
            self.BE_list = np.load(s.results_folder + 'LCOE/' + 'BE_list_{0}.npy'.format(n))
            self.SPC_c_list = np.load(s.results_folder + 'LCOE/' + 'SPC_c_list_{0}.npy'.format(n))
            self.SPC_d_list = np.load(s.results_folder + 'LCOE/' + 'SPC_d_list_{0}.npy'.format(n))
            self.SEC_list = np.load(s.results_folder + 'LCOE/' + 'SEC_list_{0}.npy'.format(n))

        if solve_LCOE:
            self.LCOE_BC_list = np.empty(n)
            self.LCOE_BE_list = np.empty(n)
            self.LCOE_SPC_c_list = np.empty(n)
            self.LCOE_SPC_d_list = np.empty(n)
            self.LCOE_SEC_list = np.empty(n)

            for i, (b, BC, BE, SPC_c, SPC_d, SEC) in tqdm(enumerate(zip(beta_list,
                                                                        self.BC_list,
                                                                        self.BE_list,
                                                                        self.SPC_c_list,
                                                                        self.SPC_d_list,
                                                                        self.SEC_list))):
                    (self.LCOE_BC_list[i],
                     self.LCOE_BE_list[i],
                     self.LCOE_SPC_c_list[i],
                     self.LCOE_SPC_d_list[i],
                     self.LCOE_SEC_list[i]) = self.calculate_costs(BC, BE,
                                                                   SPC_c,
                                                                   SPC_d,
                                                                   SEC, prices_backup, prices_storage)
            if save_LCOE:
                np.save(s.results_folder + 'LCOE/' + 'LCOE_BC_list_{0}'.format(n), self.LCOE_BC_list)
                np.save(s.results_folder + 'LCOE/' + 'LCOE_BE_list_{0}'.format(n), self.LCOE_BE_list)
                np.save(s.results_folder + 'LCOE/' + 'LCOE_SPC_c_list_{0}'.format(n), self.LCOE_SPC_c_list)
                np.save(s.results_folder + 'LCOE/' + 'LCOE_SPC_d_list_{0}'.format(n), self.LCOE_SPC_d_list)
                np.save(s.results_folder + 'LCOE/' + 'LCOE_SEC_list_{0}'.format(n), self.LCOE_SEC_list)

        else:
            self.LCOE_BC_list = np.load(path + 'LCOE_BC_list.npy')
            self.LCOE_BE_list = np.load(path + 'LCOE_BE_list.npy')
            self.LCOE_SPC_c_list = np.load(path + 'LCOE_SPC_c_list.npy')
            self.LCOE_SPC_d_list = np.load(path + 'LCOE_SPC_d_list.npy')
            self.LCOE_SEC_list = np.load(path + 'LCOE_SEC_list.npy')


# --------------------------------------------------------------------------------------------------

L = LCOE_storage()
L.test_all(beta_list=np.linspace(0, 1.5, 16), solve_lists=True, save_lists=True, solve_LCOE=True, save_LCOE=True)
# --------------------------------------------------------------------------------------------------

def plot_timeseries():
    L.get_Sn()
    L.get_BE()
    plt.plot(L.backup_relative[18], 'r')
    plt.plot(L.Bs_n_relative[18], 'b')
    plt.plot(L.Sn_relative[18], 'g')
    plt.plot(L.curtailment_relative[18], 'c')
    plt.show()


def plot_results(n=31):
    path = s.results_folder + 'LCOE/'

    BC = np.load(path + 'BC_list_{0}.npy'.format(n))
    BE = np.load(path + 'BE_list_{0}.npy'.format(n))
    SEC = np.load(path + 'SEC_list_{0}.npy'.format(n))
    SPC_c = np.load(path + 'SPC_c_list_{0}.npy'.format(n))
    SPC_d = np.load(path + 'SPC_d_list_{0}.npy'.format(n))

    LCOE_BC = np.load(path + 'LCOE_BC_list_{0}.npy'.format(n))
    LCOE_BE = np.load(path + 'LCOE_BE_list_{0}.npy'.format(n))
    LCOE_SEC = np.load(path + 'LCOE_SEC_list_{0}.npy'.format(n))
    LCOE_SPC_c = np.load(path + 'LCOE_SPC_c_list_{0}.npy'.format(n))
    LCOE_SPC_d = np.load(path + 'LCOE_SPC_d_list_{0}.npy'.format(n))
    LCOE_SPC = LCOE_SPC_c + LCOE_SPC_d

    beta_list = np.linspace(0, 1.5, len(BC))

    backup_costs = LCOE_BC + LCOE_BE
    storage_costs = LCOE_SPC_c + LCOE_SPC_d + LCOE_SEC
    total_costs = backup_costs + storage_costs

    # Plotting stacked plot of all LCOE
    fig1, (ax1) = plt.subplots(1, 1, sharex=True)
    l = ('LCOE BC', 'LCOE BE', 'LCOE SPC_c', 'LCOE SPC_d', 'LCOE SEC')
    ax1.stackplot(beta_list, LCOE_BC, LCOE_BE, LCOE_SPC_c, LCOE_SPC_d, LCOE_SEC, labels=l)
    ax1.set_xlim([0, 1.5])
    ax1.set_ylim([0, 30])
    ax1.set_ylabel(r'$€/MWh$')
    ax1.set_xlabel(r'$\beta^B$')
    ax1.legend()
    fig1.savefig(path + 'stacked_LCOE_{0}.pdf'.format(n), bbox_inches='tight')

    # Plots of power and energy
    fig2, (ax21, ax22, ax23, ax24, ax25) = plt.subplots(5, 1, sharex=True)
    ax21.plot(beta_list, BC, label = 'Backup Capacity')
    ax22.plot(beta_list, BE, label = 'Backup Energy')
    ax23.plot(beta_list, SPC_c, label = 'Storage charge power')
    ax24.plot(beta_list, SPC_d, label = 'Storage discharge power')
    ax25.plot(beta_list, SEC, label = 'Storage Capaticy')
    ax25.set_xlabel(r'$\beta^B$')
    ax22.set_xlim([0, 1.5])
    ax21.legend()
    ax22.legend()
    ax23.legend()
    ax24.legend()
    ax25.legend()
    fig2.savefig(path + 'BC_BE_SPC_SEC_{0}.pdf'.format(n), bbox_inches='tight')

    # Plots of individual costs.
    fig3, (ax31, ax32, ax33, ax34, ax35) = plt.subplots(5, 1, sharex=True, sharey=True)
    ax31.plot(beta_list, LCOE_BC, label = 'LCOE Backup Capacity')
    ax32.plot(beta_list, LCOE_BE, label = 'LCOE Backup Energy')
    ax33.plot(beta_list, LCOE_SPC_c, label = 'LCOE Charge Storage Power')
    ax34.plot(beta_list, LCOE_SPC_d, label = 'LCOE Disharge Storage Power')
    ax35.plot(beta_list, LCOE_SEC, label = 'LCOE Storage Capaticy')
    ax35.set_xlabel(r'$\beta^B$')
    fig3.text(0.04, 0.5, '$€/MWh$', va='center', rotation='vertical')
    ax35.set_ylim([0, 30])
    ax35.set_xlim([0, 1.5])
    ax31.legend()
    ax32.legend()
    ax33.legend()
    ax34.legend()
    ax35.legend()
    fig3.savefig(path + 'LCOE_BC_BE_SPC_SEC_{0}.pdf'.format(n), bbox_inches='tight')

    plt.show()

# plot_results()
