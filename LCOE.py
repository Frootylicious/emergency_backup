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
#         self.eta_store = 1.00
#         self.eta_dispatch = 1.00
        self.eta_store = 0.75
        self.eta_dispatch = 0.58
        self.beta_capacity = 0.50

        # Costs ------------------------------------------------------------------------------------

        self.backup_price = {
                             'CapCostPower': 0.9, # €/W
                             'OpExFixed': 4.5, # €/kW/year
                             'OpExVariable': 56, # €/MWh/year
                            }
        self.storage_price = {
                              'CapCostPower': 737,
                              'O&M': 12.2,
                              'CapCostEnergy': 11.2,
                             }



        self.annuity_n = 20
        self.annuity_r = 0.07

        self.DOLLAR_TO_EURO = 0.7532

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

        # States of solving ------------------------------------------------------------------------
        self.solved_S_n = False
        self.solved_BC = False
        self.solved_BE = False
        self.solved_K_PS = False
        self.solved_K_ES = False

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
        if not self.solved_S_n:
#             print('Solving storage...')

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
            self.solved_S_n = True

    def get_BC(self):
        '''
        Backup Capacity (BC)  [MW] which is \beta * <L_n> which is summed for each country
        '''
        if not self.solved_BC:
            self.BC = np.sum([self.beta_capacity * L for L in self.avg_L_n])
            self.solved_BC = True

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
        if not self.solved_BE:
            self.get_S_n()
            self.BE = np.sum([b * l for b, l in zip(self.Bs_n_relative, self.avg_L_n)])
            self.solved_BE = True

    def get_K_ES(self):
        '''
        Storage Energy (SE) [MWh] the summed storage for all nodes.
        '''
        if not self.solved_S_n:
            self.get_S_n()
        if not self.solved_K_ES:
            self.K_ES = -np.min(self.S_all)
            self.solved_K_ES = True

    def get_K_PS(self):
        '''
        Max charging and discharging rate.
        Max and min of diff of the filling level of the storage.
        '''
        if not self.solved_S_n:
            self.get_S_n()
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

        # Backup capacity
        # Storage costs (DAVID) --------------------------------------------------------------------
        storage_power = np.max((np.abs(self.K_PS_charge), np.abs(self.K_PS_discharge)))
        storage_capacity = self.K_ES

        
        # Backup costs (LEON) ----------------------------------------------------------------------
        # Need Backup Energy in  MWh/year
        BE_per_year = self.BE / 8
        # Backup capacity in MW
        BC = self.BC
        # Costs:
        BE_costs = BE_per_year*asset_CCGT['OpExVariable'] * annualizationFactor(asset_CCGT['Lifetime'])
        BC_costs = self.BC*(asset_CCGT['CapExFixed'] * 1e6 + asset_CCGT['OpExFixed'] * 1e3 * annualizationFactor(asset_CCGT['Lifetime']))

        # Cost of electrolysis, fuel cells and steel tanks.
        SP_costs = storage_power * (737 + annualizationFactor(20) * 12.2) * 1e3 * self.DOLLAR_TO_EURO
        SE_costs = storage_capacity * 11.2 * 1e3 * self.DOLLAR_TO_EURO




        scalingFactor_20 = self.all_load / 8 *annualizationFactor(20)
        
        LCOE_BE = BE_costs / scalingFactor_20
        LCOE_BC = BC_costs / scalingFactor_20

        LCOE_SP = SP_costs / scalingFactor_20
        LCOE_SE = SE_costs / scalingFactor_20
        
        self.TOTAL_LCOE_B = LCOE_BE + LCOE_BC
        self.TOTAL_LCOE_S = LCOE_SE + LCOE_SE

        self.TOTAL_LCOE = self.TOTAL_LCOE_B + self.TOTAL_LCOE_S
#         print('LCOE BACKUP: ')
#         print(t.convert_to_exp_notation(self.TOTAL_LCOE_B))
#         print('LCOE STORAGE: ')
#         print(t.convert_to_exp_notation(self.TOTAL_LCOE_S))
#         print('TOTAL LCOE: ')
#         print(t.convert_to_exp_notation(self.TOTAL_LCOE))


    def test(self, beta_capacity=0.50):
        if beta_capacity != self.beta_capacity:
            self.solved_S_n = False
            self.solved_BC = False
            self.solved_BE = False
            self.solved_K_PS = False
            self.solved_K_ES = False
            self.beta_capacity=beta_capacity
            self.calculate_storage()
        print('----------- Beta is: {0} -----------'.format(self.beta_capacity))
#         print('AVERAGE LOAD EU')
#         print(t.convert_to_exp_notation(self.avg_L_EU) + ' MWh')
#         print('BACKUP CAPACITY')
#         print(t.convert_to_exp_notation(self.BC) + ' MWh')
#         print('BACKUP ENERGY')
#         print(t.convert_to_exp_notation(self.BE) + ' MWh')
#         print('EMERGENCY BACKUP CAPACITY')
#         print(t.convert_to_exp_notation(self.K_ES) + ' MWh')
#         print('EMERGENCY BACKUP MAX CHARGING PER HOUR')
#         print(t.convert_to_exp_notation(self.K_PS_charge) + ' MWh')
#         print('EMERGENCY BACKUP MAX DISCHARGING PER HOUR')
#         print(t.convert_to_exp_notation(self.K_PS_discharge) + ' MWh')
#         print('---------------------------------------')
        return

    def test_BE(self):
        for i, b in enumerate(np.linspace(0, 1.25, 26)):
            self.test(beta_capacity=b)


    def test_all(self, beta_list=np.linspace(0.00, 1.50, 1)):
        self.beta_list = beta_list
        n = len(beta_list)
        self.BC_list = np.empty(n)
        self.BE_list = np.empty(n)
        self.SC_list = np.empty(n)
        self.S_charge_list = np.empty(n)

        self.backup_costs = np.empty(n)
        self.storage_costs = np.empty(n)
        self.total_costs = np.empty(n)

        for i, b in tqdm(enumerate(beta_list)):
            self.test(beta_capacity=b)

            self.BC_list[i] = self.BC
            self.BE_list[i] = self.BE
            self.SC_list[i] = self.K_ES
            self.S_charge_list[i] = np.max((np.abs(self.K_PS_charge), np.abs(self.K_PS_discharge)))

            self.calculate_costs()
            self.backup_costs[i] = self.TOTAL_LCOE_B
            self.storage_costs[i] = self.TOTAL_LCOE_S
            self.total_costs[i] = self.TOTAL_LCOE

        np.save(s.results_folder + 'LCOE/' + 'BC_list', self.BC_list)
        np.save(s.results_folder + 'LCOE/' + 'BE_list', self.BE_list)
        np.save(s.results_folder + 'LCOE/' + 'SC_list', self.SC_list)
        np.save(s.results_folder + 'LCOE/' + 'S_charge_list', self.S_charge_list)

        np.save(s.results_folder + 'LCOE/' + 'backup_costs', self.backup_costs)
        np.save(s.results_folder + 'LCOE/' + 'storage_costs', self.storage_costs)
        np.save(s.results_folder + 'LCOE/' + 'total_costs', self.total_costs)
            
#         fig, (ax) = plt.subplots(1, 1, sharex=True)
#         ax.plot(beta_list, self.total_costs, label='Total Costs')
#         ax.plot(beta_list, self.backup_costs, label='Backup Costs')
#         ax.plot(beta_list, self.storage_costs, label='Storage Costs')
#         ax.set_ylabel(r'$€/MWh$')
#         ax.set_xlabel(r'$\beta^B$')
#         ax.legend()
#         plt.show()
        




# --------------------------------------------------------------------------------------------------

L = LCOE_storage()
# --------------------------------------------------------------------------------------------------

def plotthings(calc=False):
    if calc:
        print('Calculating....')
        L.get_S_n()
        L.get_BE()
    plt.plot(L.balancing_relative[18], 'r')
    plt.plot(L.Bs_n_relative[18], 'b')
    plt.plot(L.S_n_relative[18], 'g')
    plt.plot(L.curtailment_relative[18], 'c')
    plt.show()
