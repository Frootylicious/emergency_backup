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
DOLLAR_TO_EURO = 0.89114646

#####
## Cost assumptions: // Source: Rolando PHD thesis, table 4.1, page 109. Emil's thesis. 
asset_CCGT = {
    'Name': 'CCGT backup',
    'CapExFixed': 0.9, #Euros/W
    'OpExFixed': 4.5, #Euros/kW/year
    'OpExVariable': 56.0, #Euros/MWh
    'Lifetime': 30 #years
    }

H2 = {
      'Capital cost per energy storage':     28.1,     #$/kWh
      'Lifetime of energy equipment':        20,       #years 
      'Capital power cost':                1683,       #$/kW capacity
      'O&M cost per unit of power capacity': 27.5,     #$/kW/year
      'O&M net present cost for 20 years':  206,       #$/kW
      'Lifetime power equipment':            20,       #years
      'Energy cost for 20 years':            28.1,     #$/kWh
      'Power cost for 20 years':           1889,       #$/kW
      'Round trip efficiency':                0.438,   #Fraction
      'Storage loss over time':               1.50e-8, #Fraction lost per hour
      }

asset_H2 = {
      'Name': 'H2 emergency backup',
      'CapExFixed': H2['Capital power cost'] / 1000 * DOLLAR_TO_EURO, # €/W
      'OpExFixed' : H2['O&M cost per unit of power capacity'] * DOLLAR_TO_EURO, # €/kW/year
      'OpExVariable': H2['Capital cost per energy storage'] / 1000 * DOLLAR_TO_EURO
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

def cost_ES(N):
    return

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
