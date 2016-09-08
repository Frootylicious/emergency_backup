import numpy as np
from solve_networks.solve_multiple_networks import solve_multiple_networks

alpha_list = [0.80]
# gamma_list = [1.00]
# beta_list = [1.00]
alpha_list = np.linspace(0, 1, 11)
beta_list = (0, np.inf)
network_solving = solve_multiple_networks(alpha_list=alpha_list, 
                                          gamma_list=gamma_list,
                                          beta_list=beta_list)


