import numpy as np
from scipy import optimize

class QQPlot(object):
    def __init__(self, load_path, save_path):
        "docstring"
        

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
