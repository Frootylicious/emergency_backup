 # -*- coding: utf8 -*-
from __future__ import division
import numpy as np
import settings as s
import regions.classes as cl
import aurespf.solvers as au
import aurespf.DCsolvers as dc
import regions.tools as to
import os


class Data():
    '''
    Class to hold the Nodes and Flow objects.

    If no arguments are passed, a network with a = 0, gamma = 1 is solved in
    the unconstrained and synchronized flowscheme.

    Arguments:
        load/save: Whether network should be loaded or saved.
        a: alpha (mixing between wind/solar)
        g: gamma (renewables  penetration)
        b: beta  (scaling factor on constraints)
        filename: the prefix name for the Nodes and Flow objects.
        DC: Use Magnus' DC-solver or the AURESPF.
        constrained: Whether the network is constrained or unconstrained.
        save: Whether the network should save the Nodes and Flow objects.
    '''

    def __init__(self, a=0.80, g=1.00, b=1.00,
                 mode='square',
                 DC=True,
                 constrained=True,
                 load=False,
                 save_F=False):
        self.a = a
        self.g = g
        self.b = b
        self.mode = mode
        self.DC = DC
        self.save_F = save_F
        self.constrained = constrained
        str_constrained = 'c' if constrained else 'u'
        str_lin_syn = 's' if 'square' in mode else 'l'
        self.nodes_name = s.nodes_name.format(c=str_constrained, f=str_lin_syn, a=a, g=g, b=b)
        self.path = s.nodes_folder
        self.fullname = self.path + self.nodes_name + '_N.npz'
        # Listing all filenames and link names ---------------------------------
        self.files = s.files
        self.link_list = s.link_list

        # Should the network be solved or loaded from file.
        if not os.path.isfile(self.fullname):
            print('Network not solved - solving and saving...')
            self.solve_network()
        else:
            print('Network already solved.')

        if not os.path.exists(s.results_folder):
            os.makedirs(s.results_folder)


    def find_country(self, country='DK'):
        # Returns the county's index number in the list.
        return(self.files.index(country + '.npz'))

    def find_links(self, country='DK'):
        # Get a list of links containing a specific country
        return [x for x in self.link_list if country in x]

    def find_link(self, link_str):
        # Get the specific link index
        try:
            return(self.link_list.index(link_str))
        except:
            print('Link "' + link_str + '''" doesn't exist''')

    ## SOLVE -------------------------------------------------------------------
    def solve_network(self):
        '''
        Solving the network.

        Differs significantly whether the network is constrained or not.
        '''
        N_name = self.fullname
        F_name = self.fullname.replace('N', 'F')
        self.N = cl.Nodes(admat='./settings/eadmat.txt',
                          path=s.iset_folder,
                          prefix="ISET_country_",
                          files=s.files,
                          alphas=self.a,
                          gammas=self.g)
        msg = ('{0} {1}-network with mode = "{2}"\nALPHA = {3:.2f}, GAMMA = {4:.2f}'
                ', BETA = {5:.2f}\n')


        # Solving a constrained network. To find h0, a constrained network with
        # same alpha and gamma are needed. Therefore we check for for a file in
        # data/copperflows/ with the desired values. If it's not there, it is
        # calculated and saved.
        if self.constrained:
            if not os.path.exists(s.copper_folder):
                os.makedirs(s.copper_folder)
            mode = 'square' if 'square' in self.mode else 'linear'
            # File naming for the unconstrained files.
            if not os.path.isfile(s.copper_fullname.format(self.a, self.g)):
                print("No copperflow file '{0}' - solving it and"
                        " saving...").format(s.copper_name.format(self.a, self.g))
                msgCopper = ('unconstrained DC-network with mode = "{0}"\nALPHA ='
                        ' {1:.2f}, GAMMA = {2:.2f}').format(mode, self.a, self.g)
                M_copperflows, F_copperflows = dc.DC_solve(self.N,
                                                           mode='copper ' + mode,
                                                           msg=msgCopper)
                np.save(s.copper_fullname.format(self.a, self.g), F_copperflows)
                print('Saved copper flows to file:{0}'.format(s.copper_name.format(self.a,
                                                                                 self.g)))
            else:
                print("Found copperflow file - using for constrained flow")

            # Calculating the 99 % quantile. This function from tools takes a
            # quantile and a filename for the unconstrained flow with same alpha
            # and gamma values.
            h0 = to.get_quant_caps(quant=0.99, filename=s.copper_fullname.format(self.a, self.g))
            msg_constrained = msg.format('constrained',
                                         'DC',
                                         mode,
                                         self.a,
                                         self.g,
                                         self.b)
            self.M, self.F = dc.DC_solve(self.N, h0=h0, b=self.b, mode=mode,
                                         msg=msg_constrained)


        # Solving an unconstrained network with Magnus' DC-solver.
        elif not self.constrained and self.DC:
            self.M, self.F = dc.DC_solve(self.N,
                                         mode=self.mode,
                                         msg=msg.format('unconstrained',
                                                        'DC',
                                                        self.mode,
                                                        self.a,
                                                        self.g,
                                                        self.b))
        # Solving an unconstrained network with AURESPF solver.
        else:
            self.M, self.F = au.solve(self.N, mode=self.mode,
                                      msg=msg.format('unconstrained',
                                                        'non-DC',
                                                        self.mode,
                                                        self.a,
                                                        self.g,
                                                        self.b))

        # Checking if results-folder exists. Create it if not.
        if not os.path.exists(s.results_folder):
            os.makedirs(s.results_folder)
        if not os.path.exists(s.nodes_folder):
            os.makedirs(s.nodes_folder)
        if not os.path.exists(s.links_folder) and self.save_F:
            os.makedirs(s.links_folder)
        self.M.save_nodes(filename=self.nodes_name + '_N.npz', path=s.nodes_folder)
        if self.save_F:
            np.savez_compressed(F_name, self.F)






    ## LOAD --------------------------------------------------------------------
    # Loading network with parameters set in __init__.
#     def load_network(self):
#         F_name = 'results/' + self.filename + '_F.npz'
#         N_name = self.filename
#         self.N = cl.Nodes(load_filename=N_name,
#                           files=self.files,
#                           path='./data/',
#                           prefix='ISET_country_')
#         self.links = np.load(F_name)
#         self.F = self.links.f.arr_0

if __name__ == '__main__':
    B = Data(DC=False, constrained=False)
