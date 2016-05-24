from solve_networks.solve_multiple_networks import solve_multiple_networks

alpha_list = [0.80]
gamma_list = [1.00]
beta_list = [1.00]
network_solving = solve_multiple_networks(alpha_list=alpha_list, 
                                          gamma_list=gamma_list,
                                          beta_list=beta_list)


