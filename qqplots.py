import numpy as np
from scipy import optimize

class QQPlot(object):
    def __init__(self, load_path, save_path):
        "docstring"
        self.load_path = load_path
        self.save_path = save_path
        self.plots = {}
        self.countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                          'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                          'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
        self.country_dict = dict(zip(self.countries, list(range(len(self.countries)))))

    ###############################
    ####### Private methods #######
    ###############################       
    def _quantile(self, quantile, dataset, cutzeros=True):
        """
        Takes a list of numbers, converts it to a list without zeros
        and returns the value of the 'quantile' quantile.
        """
        if cutzeros:
            # Removing zeros
            dataset = dataset[np.nonzero(dataset)]
        # Convert to numpy histogram
        hist, bin_edges = np.histogram(dataset, bins = 10000, normed = True)
        dif = np.diff(bin_edges)[0]
        q = 0
        for index, val in enumerate(reversed(hist)):
            q += val*dif
            if q > 1 - float(quantile)/100:
                #print 'Found %.3f quantile' % (1 - q)
                return bin_edges[-index]

    def _load_backup_timeseries(self, dataTouple):
        """
        Takes a touple of the form (country, constraint, flow scheme, alpha, gamma, beta)
        and loads the corresponding timeseries from load_path.

        Assumes files with the names on the form c/u_l/s_a{}_g{}_b{}.npz,
        where c means constrained, u means unconstrained, l means linear, s means synchronized
        and 
        """
        country = self.country_dict[dataTouple[0]]
        return np.load('%s%s_%s_a%.2f_g%.2f_b%.2f.npz' \
                       % (load_path,
                          dataTouple[1],
                          dataTouple[2],
                          dataTouple[3],
                          dataTouple[4],
                          dataTouple[4]))['arr_0'][country]


    ##############################
    ####### Public methods #######
    ##############################
    def addPlot(self, dataTouple):
        """
        Takes a touple of the form (country, constraint, flow scheme, alpha, gamma, beta)
        and adds it to the plots-list
        """
        self.plots[dataTouple] = self._load_backup_timeseries(dataTouple)

