 # -*- coding: utf8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import regions.classes as cl
import aurespf.solvers as au
import aurespf.DCsolvers as dc
import regions.tools as to
import os
import matplotlib
matplotlib.style.use('seaborn-dark')


class Data():
    '''
    Class to hold the Nodes and node objects.

    If no arguments are passed, a network with a = 0, gamma = 1 is solved in
    the unconstrained and synchronized flowscheme.
    '''

    def __init__(self, load=True, solve=False, a=0.80, g=1.00,
                 mode='copper square verbose',
                 filename='test_result',
                 DC=False,
                 constrained=False,
                 b=1.00):
        self.a = a
        self.g = g
        self.mode = mode
        self.filename = filename
        self.DC = DC
        self.constrained = constrained
        self.b = b

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
        F_name = 'results/' + self.filename + '_F.npz'
        self.N = cl.Nodes(admat='./settings/eadmat.txt',
                          path='./data/',
                          prefix="ISET_country_",
                          files=self.files,
                          load_filename=None,
                          full_load=False,
                          alphas=self.a,
                          gammas=self.g)
        msg = ('{0} {1}-network with mode = {2}\nALPHA = {3:.2f}, GAMMA = {4:.2f}'
                ', BETA = {5:.2f}\n')


        if self.constrained:
            # Need to calculate h0. If the file is not calculated - do it.
            mode = ['square' if 'square' in self.mode else 'linear'][0]
            copper_file = 'data/copperflows/copperflow_a{0:.2f}_g{1:.2f}.npy'
            if not os.path.isfile(copper_file.format(self.a, self.g)):
                print("Couldn't find file '{0}' - solving it first and"
                        " saving...").format(copper_file.format(self.a, self.g))
                msgCopper = ('unconstrained DC-network with mode = "{0}"\nALPHA ='
                        ' {1:.2f}, GAMMA = {2:.2f}').format(mode, self.a, self.g)
                M_copperflows, F_copperflows = dc.DC_solve(self.N,
                                                           mode=mode,
                                                           msg=msgCopper)
                np.save(copper_file.format(self.a, self.g),
                                    F_copperflows)
                print('Saved copper flows to file:{0}'.format(copper_file.format(self.a, self.g)))

            h0 = to.get_quant_caps(filename=copper_file.format(self.a, self.g))
            msg_constrained = msg.format('constrained',
                                         'DC',
                                         mode,
                                         self.a,
                                         self.g,
                                         self.b)
            self.M, self.F = dc.DC_solve(self.N, h0=h0, b=self.b, mode=mode,
                                         msg=msg_constrained)

        elif not self.constrained and self.DC:
            self.M, self.F = dc.DC_solve(self.N,
                                         mode=self.mode,
                                         msg=msg.format('unconstrained',
                                                        'DC',
                                                        self.mode,
                                                        self.a,
                                                        self.g,
                                                        self.b))

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
        self.M.save_nodes(filename=self.filename + '_N')
        np.savez(F_name, self.F)

    ## LOAD --------------------------------------------------------------------
    def load_network(self):
        F_name = 'results/' + self.filename + '_F.npz'
        N_name = self.filename + '_N.npz'
        self.N = cl.Nodes(load_filename=N_name,
                          files=self.files,
                          path='./data/',
                          prefix='ISET_country_')
        self.links = np.load(F_name)
        self.F = self.links.f.arr_0
