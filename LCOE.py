# -*- coding: utf-8 -*-

#############################################################
## Cost tools from Mads Raunbak and original tools.py file ##
#############################################################

################################################################################################
## Corrected for Jonas' new regions implementation where fx mismatch has transposed dimension ##
################################################################################################

import numpy as np
import regions.tools as tools
HOURS_PER_YEAR = 8760

#####
## Cost assumptions: // Source: Rolando PHD thesis, table 4.1, page 109. Emil's thesis. 
asset_CCGT = {
    'Name': 'CCGT backup',
    'CapExFixed': 0.9, #Euros/W
    'OpExFixed': 4.5, #Euros/kW/year
    'OpExVariable': 56.0, #Euros/MWh/year
    'Lifetime': 30 #years
    }

#####
## Annualization factor:
def annualizationFactor(lifetime, r=4.0): 
    """Lifetime in years and r = rate in percent."""
    if r==0: return lifetime
    return (1-(1+(r/100.0))**-lifetime)/(r/100.0)
    
#####
## Calculating BE, BC, TC, and wind and solar capacity.
def get_BE(N):
    """Total annual BE. In MWh per year."""
    return sum([sum(n.balancing) for n in N]) / N.number_of_hours * HOURS_PER_YEAR
    
def get_BC(N):
    """Total BC. In MW."""
    return sum([tools.get_q(n.balancing, 0.99) for n in N])
 
    
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
    scalingFactor_25 = sum([sum(n.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(25)
    scalingFactor_30 = sum([sum(n.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(30)
    scalingFactor_40 = sum([sum(n.load) for n in N])/N.number_of_hours*HOURS_PER_YEAR*annualizationFactor(40)
    
    LCOE_BE = cost_BE(N) / scalingFactor_30
    LCOE_BC = cost_BC(N) / scalingFactor_30
    
    TOTAL_LCOE = LCOE_BE + LCOE_BC

    return TOTAL_LCOE,LCOE_BE,LCOE_BC
