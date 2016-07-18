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
        self.Bc_price = 737
        self.Be_price = 737
        self.Ks_power_price = 737 # $/kW capacity - Capital power cost
        self.Ks_energy_price = 11.2 # $/kWh - Capital cost per energy storage (cost of steel tanks) 
        self.Kc_price = 12.2 # $/kW/year - O&M cost per unit of power capacity (equipment for electrolysis/fuel cells)

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

        self.S_n : all countries' storage filling level timeseries.
        self.S_all : a storage filling level timeseries for the combined Europe.
        '''
        if not self.solved_S_n:
            print('Solving storage...')
            S_n = [t.storage_size_relative(b,
                                           self.beta_capacity,
                                           eta_in=self.eta_store,
                                           eta_out=self.eta_dispatch)[1] * L for b, L in zip(self.balancing_relative, self.avg_L_n)]
            self.S_n = np.array(S_n)
            self.S_all = np.sum(self.S_n, axis=0)
            self.S_n_relative = np.array([s / l for s, l in zip(self.S_n, self.avg_L_n)])
            self.solved_S_n = True

    def get_BC(self):
        '''
        Backup Capacity (BC)  [MW] which is \beta * <L_n> which is summed for each country
        '''
        if not self.solved_BC:
            print('Solving backup capacity...')
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
        if not self.solved_S_n:
            self.get_S_n()
        if not self.solved_BE:
            print('Solving backup energy...')
            k = self.beta_capacity
            self.balancing_with_storage = np.copy(self.balancing_relative)
            # First we set all entries higher than our chosen capacity equal to the capacity.
            self.balancing_with_storage[self.balancing_with_storage > k] = k

            # Looping over each country
            for i, (balancing, storage, curtailment, l) in enumerate(zip(self.balancing_with_storage,
                                                                         self.S_n_relative,
                                                                         self.curtailment_relative,
                                                                         self.avg_L_n)):
                # Looping over each timestep
                for j, (b, s, c) in enumerate(zip(balancing, storage, curtailment)):
                    # If storage is not full
                    if np.abs(s) > 0:
                        # If there is curtailment present.
                        if c > 0:
                            # If the storage depletion is greater than the curtailment.
                            if np.abs(s) > c:
                                # Subtract the curtailment or set to 0
                                self.balancing_with_storage[i, j] = np.max((0, (b - c)))
                                b = np.max((0, (b - c)))
                            # If the storage depletion is smaller than the curtailment.
                            else:
                                # Subtract the storage depletion or set to 0.
                                self.balancing_with_storage[i, j] = np.max((0, (b - np.abs(s))))
                                b = np.max((0, (b - np.abs(s))))
                        # If the storage depletion + backup is smaller than the capacity
                        if np.abs(s) + b < k:
                            # Add the storage to the backup.
                            self.balancing_with_storage[i, j] += (np.abs(s))
                        else:
                            # Set backup to the maximum capacity
                            self.balancing_with_storage[i, j] = k
            self.BE = np.sum([b * L for b, L in zip(self.balancing_with_storage, self.avg_L_n)])
            self.solved_BE = True

    def get_K_ES(self):
        '''
        Storage Energy (SE) [MWh] the summed storage for all nodes.
        '''
        if not self.solved_S_n:
            self.get_S_n()
        if not self.solved_K_ES:
            print('Solving storage energy...')
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
        self.Be_cost = self.Be_price * self.DOLLAR_TO_EURO
        self.Bc_cost = self.Bc_price * self.DOLLAR_TO_EURO
        self.Ks_cost = self.Ks_price * self.DOLLAR_TO_EURO
        self.Kc_cost = self.Kc_price * self.DOLLAR_TO_EURO


    def test(self, beta_capacity=1.00):
        if beta_capacity != self.beta_capacity:
            self.solved_S_n = False
            self.solved_BC = False
            self.solved_BE = False
            self.solved_K_PS = False
            self.solved_K_ES = False
            self.beta_capacity=beta_capacity
            self.calculate_storage()
        print('----------- Beta is: {:.2f} -----------'.format(self.beta_capacity))
        print('AVERAGE LOAD EU')
        print(t.convert_to_exp_notation(self.avg_L_EU) + ' MWh')
        print('BACKUP CAPACITY')
        print(t.convert_to_exp_notation(self.BC) + ' MWh')
        print('BACKUP ENERGY')
        print(t.convert_to_exp_notation(self.BE) + ' MWh')
        print('EMERGENCY BACKUP CAPACITY')
        print(t.convert_to_exp_notation(self.K_ES) + ' MWh')
        print('EMERGENCY BACKUP MAX CHARGING PER HOUR')
        print(t.convert_to_exp_notation(self.K_PS_charge) + ' MWh')
        print('EMERGENCY BACKUP MAX DISCHARGING PER HOUR')
        print(t.convert_to_exp_notation(self.K_PS_discharge) + ' MWh')
        print('---------------------------------------')
        return

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

## Cost assumptions: // Source: Rolando PHD thesis, table 4.1, page 109. Emil's thesis.
asset_CCGT = {
    'Name': 'CCGT backup',
    'CapExFixed': 0.9, #Euros/W
    'OpExFixed': 4.5, #Euros/kW/year
    'OpExVariable': 56.0, #Euros/MWh
    'Lifetime': 30 #years
    }

asset_H2 = {
        'CapExFixed': 737, # $/kW capacity
        'OpExFixed': 12.2, # $/kW/year
        'OpExVariable': 11.2, # $/kWh
        }

#####
## The five essential cost terms from Emil's implementation:
def cost_BE(N):
    """Cost of BE is variable part of CCGT."""
    BE = get_BE(N)
    return BE*asset_CCGT['OpExVariable']*annualizationFactor(asset_CCGT['Lifetime'])

def cost_BC(N):
    """Cost of BC is fixed part of CCGT."""
    BC = get_BC(N)
    return BC*(asset_CCGT['CapExFixed']*1e6 + asset_CCGT['OpExFixed']*1e3*annualizationFactor(asset_CCGT['Lifetime']))


#####
## Total energy consumption: Used as scaling for the LCOE:
def total_annual_energy_consumption(N):
    return np.sum([n.mean for n in N])*HOURS_PER_YEAR

#####
## LCOE: With NEW definition: Scaling term depends on the lifetime of the investment:
def get_LCOE(N):
    scalingFactor_25 = np.sum(N.f.load) / N.f.nhours[0] * HOURS_PER_YEAR * annualizationFactor(25)
    scalingFactor_30 = np.sum(N.f.load) / N.f.nhours[0] * HOURS_PER_YEAR * annualizationFactor(30)
    scalingFactor_40 = np.sum(N.f.load) / N.f.nhours[0] * HOURS_PER_YEAR * annualizationFactor(40)
#     scalingFactor_25 = sum([sum(n.f.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(25)
#     scalingFactor_30 = sum([sum(n.f.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(30)
#     scalingFactor_40 = sum([sum(n.f.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(40)

    LCOE_BE = cost_BE(N) / scalingFactor_30
    LCOE_BC = cost_BC(N) / scalingFactor_30

    TOTAL_LCOE = LCOE_BE + LCOE_BC

    return TOTAL_LCOE,LCOE_BE,LCOE_BC

L = LCOE_storage()
# --------------------------------------------------------------------------------------------------

def plotthings(calc=False):
    if calc:
        print('Calculating....')
        L.get_S_n()
        L.get_BE()
    plt.plot(L.balancing_relative[18], 'r')
    plt.plot(L.balancing_with_storage[18], 'b')
    plt.plot(L.S_n_relative[18], 'g')
    plt.plot(L.curtailment_relative[18], 'c')
    plt.show()
