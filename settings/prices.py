from collections import namedtuple

DOLLAR_TO_EURO = 0.7532 # David

PriceStorage = namedtuple('PriceStorage', ['Name',
                                           'CapPowerCharge',
                                           'CapPowerDischarge',
                                           'OMPower',
                                           'EfficiencyCharge',
                                           'EfficiencyDischarge',
                                           'LifetimeCharge',
                                           'LifetimeDischarge',
                                           'CapStorage',
                                           'LifetimeStorage'])

PriceBackup = namedtuple('PriceBackup', ['CapExFixed',
                                         'OpExFixed',
                                         'OpExVariable',
                                         'Lifetime'])

# prices_backup_leon = {
#                      'Name': 'CCGT backup',
#                      'CapExFixed': 0.9, #Euros/W
#                      'OpExFixed': 4.5, #Euros/kW/year
#                      'OpExVariable': 56.0, #Euros/MWh/year
#                      'Lifetime': 30 #years
#                      }
#
# prices_storage_david = {
#                        'Name': 'David/Budischak storage',
#                        'Cap_Power': 737 * DOLLAR_TO_EURO, # €/kW capacity (was $/kW)
#                        'O&M_Power': 12.2 * DOLLAR_TO_EURO, # €/kW/year
#                        'Cap_Storage': 11.2 * DOLLAR_TO_EURO,# €/kWH
#                        'Lifetime': 20
#                        }
#
# prices_storage_bussar = {
#                         'Name': 'Bussar storage',
#                         'Cap_Power_Charge': 300, # €/kW capacity for charger
#                         'Cap_Power_Discharge': 400, # €/kW capacity for discharger
#                         'Capacity_Energy': 0.3, # €/kWh
#                         'Lifetime_Charger': 15,
#                         'Lifetime_Discharger': 25,
#                         'Lifetime_Storage': 40,
#                         'Efficiency_Charger': 0.80,
#                         'Efficiency_Discharger': 0.62
#                         }
#
# prices_storage_scholz = {
#                         'Name': 'Scholz Storage',
#                         'Cap_Power': 900, # €/kW capacity
#                         'O&M_Power': 0.03 * 900,
#                         'Efficiency_Charger': 0.70,
#                         'Efficiency_Discharger': 0.57,
#                         'Lifetime_Charger': 15,
#                         'Lifetime_Discharger': 15,
#                         'Cap_Storage': 1, # €/kWh
#                         'Lifetime_Storage': 30,
#                         }

prices_backup_leon = PriceBackup(CapExFixed=0.9*1e6,            # €/kW
                                 OpExFixed=4.5,                 # €/kW/year
                                 OpExVariable=56,               # €/kWh/year
                                 Lifetime=30)                   # years

prices_storage_david = PriceStorage(Name='David',
                                    CapPowerCharge=737,         # €/kW
                                    CapPowerDischarge=0,        # €/kW
                                    OMPower=12.2,               # €/kW/year
                                    EfficiencyCharge=0.75,      #
                                    EfficiencyDischarge=0.58,   #
                                    LifetimeCharge=20,          # years
                                    LifetimeDischarge=0,        # years
                                    CapStorage=11.2,            # €/kWh
                                    LifetimeStorage=20)         # years

prices_storage_bussar = PriceStorage(Name='Bussar',
                                     CapPowerCharge=300,        # €/kW
                                     CapPowerDischarge=400,     # €/kW
                                     OMPower=0,                 # €/kW/year
                                     EfficiencyCharge=0.80,     #
                                     EfficiencyDischarge=0.62,  #
                                     LifetimeCharge=15,         # years
                                     LifetimeDischarge=15,      # years
                                     CapStorage=0.3,            # €/kWh
                                     LifetimeStorage=30)        # years

prices_storage_scholz = PriceStorage(Name='Scholz',
                                     CapPowerCharge=900,        # €/kW
                                     CapPowerDischarge=0,       # €/kW
                                     OMPower=0.03 * 900,        # €/kW/year
                                     EfficiencyCharge=0.70,     #
                                     EfficiencyDischarge=0.57,  #
                                     LifetimeCharge=15,         # years
                                     LifetimeDischarge=0,       # years
                                     CapStorage=1,              # €/kWh
                                     LifetimeStorage=30)        # years
