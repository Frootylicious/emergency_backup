import settings.settings as s
import settings.tools as t
import numpy as np
import matplotlib.pyplot as plt

class LCOE():
    '''
    Calculates the Backup Capacity, Backup Energy, Storage Capacity and Storage Energy for a
    network. This is used to get a total cost of the network backup and storage.

    Args:
        beta_capacity: the backup capacity in terms of mean load
    Returns:
        BC: Backup Capacity in unit of MW
        BE: Backup Energy in unit of MWh
        SC: Storage Capacity in unit of MW
        SE: Storage Energy in unit of MWh
    '''

    def __init__(self):
        # Variables --------------------------------------------------------------------------------
        self.eta_store = 0.75
        self.eta_dispatch = 0.58
        self.beta_capacity = 0.50
        # ------------------------------------------------------------------------------------------

        # Loading the node object
        self.N = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))

        # Extracting the load of all nodes.
        self.L_EU = self.N.f.load
        self.all_load = np.sum(self.N.f.load)

        # Average load for whole EU per hour.
        self.avg_L_EU = np.mean(np.sum(self.L_EU, axis=0))
        self.balancing = self.N.f.balancing

        # Dividing each node's balancing by its average load.
        self.balancing_relative = np.array([b / np.mean(L) for b, L in zip(self.balancing, self.L_EU)])

    def calculate(self):
        self.get_S_n()
        self.get_BC()
        self.get_BE()
        self.get_K_ES()
        self.get_K_PS()


    def get_S_n(self):
        S_n = [t.storage_size_relative(b, 
                                       self.beta_capacity, 
                                       eta_in=self.eta_store, 
                                       eta_out=self.eta_dispatch)[1] * np.mean(L) for b, L in zip(self.balancing_relative, self.L_EU)]
        self.S_n = np.array(S_n)
        self.S_all = np.sum(self.S_n, axis=0)

    def get_BC(self):
        '''
        Backup Capacity (BC)  [MW] which is \beta * <L_n> which is summed for each country
        '''
        self.BC = np.sum([self.beta_capacity * np.mean(L) for L in self.L_EU])

    def get_BE(self):
        '''
        PROBABLY WRONG
        Backup Energy (BE) [MWh]  which is the summed backup energy for all nodes, when B is smaller than the BC.
        '''
        self.BE = np.sum([np.sum(b[b <= self.beta_capacity * np.mean(L)] for b, L in zip(self.balancing, self.L_EU))])

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
#         self.S_n = np.sum([t.storage_size_relative(b, self.beta_capacity, eta_in=self.eta_store,
#             eta_out=self.eta_dispatch)[1] * np.mean(L) for b, L in 
#             zip(self.balancing_relative, self.L_EU)], axis=0)

        (self.K_PS_charge, self.K_PS_discharge) = t.diff_max_min(self.S_all)


    def test(self, beta_capacity=1.00):
        self.beta_capacity=beta_capacity
        self.calculate()
        print('----------- Beta is: {:.2f} -----------'.format(self.beta_capacity))
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

#     def costs(self):
# 
#         def get_cost_K_ES():


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

L = LCOE()
