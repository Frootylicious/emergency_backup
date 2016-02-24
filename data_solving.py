 # -*- coding: utf8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import regions.classes as cl
import aurespf.solvers as au
import aurespf.DCsolvers as dc
import os
import matplotlib
matplotlib.style.use('seaborn-dark')


class Data():

    def __init__(self, load=True, solve=False, alpha=0.8, gamma=1,
                 mode='copper square verbose',
                 filename='test_result',
                 DC=False,
                 b=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.mode = mode
        self.filename = filename
        self.DC = DC
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
            # print('\nSolving network with mode "{0}"'.format(self.mode))
            # print('alpha = {0}, gamma = {1}\n'.format(self.alpha, self.gamma))
            self.solve_network()
        elif load and not solve:
            self.load_network()

    def find_country(self, country='DK'):
        return(self.files.index(country + '.npz'))

    # Get a list of links containing a specific country
    def find_links(self, country='DK'):
        return [x for x in self.link_list if country in x]

    # Get the specific link index
    def find_link(self, link_str):
        try:
            return(self.link_list.index(link_str))
        except:
            print('Link "' + link_str + '''" doesn't exist''')

    ## SOLVE -------------------------------------------------------------------
    # linear = localized
    # square = synchronized
    def solve_network(self):
        F_name = 'results/' + self.filename + '_F.npz'
        self.N = cl.Nodes(admat='./settings/eadmat.txt',
                          path='./data/',
                          prefix="ISET_country_",
                          files=self.files,
                          load_filename=None,
                          full_load=False,
                          alphas=self.alpha,
                          gammas=self.gamma)
        if self.DC:
            print('Solving DC-network with "{0}" and beta = {1}.'.format(self.mode,
                                                                      self.b))
            self.M, self.F = dc.DC_solve(self.N, mode=self.mode, b=self.b)
        else:
            print('Solving non-DC-network with "{0}".'.format(self.mode))
            self.M, self.F = au.solve(self.N, mode=self.mode)
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
        # N2 = np.load('./results/test_result.npz')
