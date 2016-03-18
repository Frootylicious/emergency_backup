# FOLDERS ------------------------------------------------------------------------------------------
iset_path = 'data/'
results_folder = 'results/'
nodes_folder = results_folder + 'N/'
links_folder = results_folder + 'F/'
figures_folder = results_folder + 'figures/'
EBC_folder = results_folder + 'emergency_capacities/'
copper_path = results_folder + 'copperflows/'

iset_prefix = 'ISET_country_'

# FILES --------------------------------------------------------------------------------------------
countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
             'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
             'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']

files = [c + '.npz' for c in countries]

link_list = ['AUT to CHE', 'AUT to CZE', 'AUT to HUN', 'AUT to DEU',
             'AUT to ITA', 'AUT to SVN', 'FIN to SWE', 'FIN to EST',
             'NLD to NOR', 'NLD to BEL', 'NLD to GBR', 'NLD to DEU',
             'BIH to HRV', 'BIH to SRB', 'FRA to BEL', 'FRA to GBR',
             'FRA to CHE', 'FRA to DEU', 'FRA to ITA', 'FRA to ESP',
             'NOR to SWE', 'NOR to DNK', 'GBR to IRL', 'POL to CZE',
             'POL to DEU', 'POL to SWE', 'POL to SVK', 'BGR to GRC',
             'BGR to ROU', 'BGR to SRB', 'GRC to ITA', 'PRT to ESP',
             'CHE to DEU', 'CHE to ITA', 'HRV to HUN', 'HRV to SRB',
             'HRV to SVN', 'ROU to HUN', 'ROU to SRB', 'CZE to DEU',
             'CZE to SVK', 'HUN to SRB', 'HUN to SVK', 'DEU to SWE',
             'DEU to DNK', 'DEU to LUX', 'SWE to DNK', 'ITA to SVN',
             'EST to LVA', 'LVA to LTU']

# FILENAMES ----------------------------------------------------------------------------------------
nodes_name = '{c}_{f}_a{a:.2f}_g{g:.2f}_b{b:.2f}'
nodes_fullname = nodes_folder + nodes_name + '_N.npz'
 
copper_name = 'copperflow_a{0:.2f}_g{1:.2f}.npy'
copper_fullname = copper_path + copper_name

EBC_name = 'EC_' + nodes_name
EBC_fullname = EBC_folder + EBC_name + '.npz'
