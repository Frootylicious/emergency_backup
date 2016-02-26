 # -*- coding: utf8 -*-
from __future__ import division
import numpy as np
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

    def __init__(self, load=True, solve=False, a=0.80, g=1.00,
                 mode='copper square verbose',
                 filename='test_result',
                 DC=False,
                 constrained=False,
                 b=1.00,
                 save=True):
        self.a = a
        self.g = g
        self.mode = mode
        self.filename = filename
        self.DC = DC
        self.constrained = constrained
        self.b = b
        self.save = save

        # Listing all filenames and link names ---------------------------------
        self.files = ['AT.npz', 'FI.npz', 'NL.npz', 'BA.npz', 'FR.npz',
                      'NO.npz', 'BE.npz', 'GB.npz', 'PL.npz', 'BG.npz',
                      'GR.npz', 'PT.npz', 'CH.npz', 'HR.npz', 'RO.npz',
                      'CZ.npz', 'HU.npz', 'RS.npz', 'DE.npz', 'IE.npz',
                      'SE.npz', 'DK.npz', 'IT.npz', 'SI.npz', 'ES.npz',
                      'LU.npz', 'SK.npz', 'EE.npz', 'LV.npz', 'LT.npz']

        self.link_list = ['AUT to CHE', 'AUT to CZE', 'AUT to HUN', 'AUT to DEU',
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

        # Should the network be solved or loaded from file.
        if solve:
            self.solve_network()
        elif load and not solve:
            self.load_network()

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
        F_name = 'results/' + self.filename + '_F.npz'
        self.N = cl.Nodes(admat='./settings/eadmat.txt',
                          path='./data/',
                          prefix="ISET_country_",
                          files=self.files,
                          load_filename=None,
                          full_load=False,
                          alphas=self.a,
                          gammas=self.g)
        msg = ('{0} {1}-network with mode = "{2}"\nALPHA = {3:.2f}, GAMMA = {4:.2f}'
                ', BETA = {5:.2f}\n')


        # Solving a constrained network. To find h0, a constrained network with
        # same alpha and gamma are needed. Therefore we check for for a file in
        # data/copperflows/ with the desired values. If it's not there, it is
        # calculated and saved.
        if self.constrained:
            mode = 'square' if 'square' in self.mode else 'linear'
            # File naming for the unconstrained files.
            copper_file = 'data/copperflows/copperflow_a{0:.2f}_g{1:.2f}.npy'
            if not os.path.isfile(copper_file.format(self.a, self.g)):
                print("Couldn't find file '{0}' - solving it and"
                        " saving...").format(copper_file.format(self.a, self.g))
                msgCopper = ('unconstrained DC-network with mode = "{0}"\nALPHA ='
                        ' {1:.2f}, GAMMA = {2:.2f}').format(mode, self.a, self.g)
                M_copperflows, F_copperflows = dc.DC_solve(self.N,
                                                           mode=mode,
                                                           msg=msgCopper)
                np.save(copper_file.format(self.a, self.g),
                                    F_copperflows)
                print('Saved copper flows to file:{0}'.format(copper_file.format(self.a,
                                                                                 self.g)))

            # Calculating the 99 % quantile. This function from tools takes a
            # quantile and a filename for the unconstrained flow with same alpha
            # and gamma values.
            h0 = to.get_quant_caps(quant=0.99, filename=copper_file.format(self.a, self.g))
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
        if not os.path.exists('results/'):
            os.makedirs('results/')
        # If variable save == True, save the solved network and flows.
        if self.save:
            self.M.save_nodes(filename=self.filename + '_N')
            np.savez(F_name, self.F)

    ## LOAD --------------------------------------------------------------------
    # Loading network with parameters set in __init__.
    def load_network(self):
        F_name = 'results/' + self.filename + '_F.npz'
        N_name = self.filename + '_N.npz'
        self.N = cl.Nodes(load_filename=N_name,
                          files=self.files,
                          path='./data/',
                          prefix='ISET_country_')
        self.links = np.load(F_name)
        self.F = self.links.f.arr_0
