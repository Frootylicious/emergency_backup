#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_backup(country, data_touple, path_to_data='results/'):
    """
    Returns the load in MWh.
    """
    # -- Unpacking the touple --
    names = ['constraint', 'flowscheme', 'alpha', 'gamma', 'beta']
    data = dict(zip(names, data_touple))
   
    countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                 'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                 'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
    country_dict = dict(zip(countries, list(range(len(countries)))))

    return np.load(path_to_data +
                   '{constraint}_{flowscheme}_a{alpha:.2f}_g{gamma:.2f}_b{beta:.2f}_N.npz'\
                   .format(**data))['balancing'][country_dict[country]]

def get_load(country, path_to_data='data/'):
    """
    Returns the load in MWh.
    """
    return np.load(path_to_data + 'ISET_country_' + country + '.npz')['L']*1000

def quantile(quantile, dataset):
    """
    Docstring for quantile.
    """
    return np.sort(dataset)[int(round(quantile*len(dataset)))]

def time_between_extreme_events(timeseries, q=0.99, ignore_hours=0):
    """
    Finds the waiting times between extreme events in a timeseries.
    
    Parameters
    ----------
    timeseries:
    The timeseries we want to analyze.

    q:
    The quantile with which we want to quantify the extreme events.
    I.e. only event rarer than q are used in the analysis.

    ignore_hours:
    The number of hours we want to ignore, if the waiting time
    is smaller than.

    Return
    ------
    np.diff(extreme_events):
    The difference between extreme events that are farther
    apart than ignore_hours.
    """
    q = quantile(q, timeseries)
    # find the hours that contain extreme events
    extreme_events = np.arange(len(timeseries))[timeseries > q]
    if ignore_hours == 0:
        return np.diff(extreme_events)
    elif ignore_hours > 0:
        extreme_events_ignore = np.zeros(len(extreme_events), dtype=bool)
        extreme_events_ignore[0] = True # We always want to include the first extreme event
        # counter used to see when we exceed the number of ignored hours
        counter = 0
        for index, dif in enumerate(np.diff(extreme_events)):
            counter += dif
            if counter > ignore_hours:
                # index + 1 works because index ranges from
                # 0 to length(extreme_events) - 1
                extreme_events_ignore[index + 1] = True
                # reset the counter
                counter = 0
        # return the waiting times between the hours that contain extreme
        # events and that are also more than ignore_hours apart.
        return np.diff(extreme_events[extreme_events_ignore])
    elif ignore_hours < 0:
        print('ignore_hours must be positive. Try again.')





    
if __name__ == '__main__':
    data_touple = ('c', 's', 0.80, 1.00, 1.00)
    DK_backup = get_backup('DK', data_touple, path_to_data='./')
    DK_load = get_load('DK', path_to_data='/home/simon/Dropbox/Root/Data/ISET/')
    evs = time_between_extreme_events(DK_backup, ignore_hours=0)
    evs_in_weeks = evs/float(24*30)
    hist, bin_edges = np.histogram(evs_in_weeks, bins=500, normed=True)
    bins = 0.5*(bin_edges[:-1] + bin_edges[1:])
    hist = hist*np.diff(bins)[0]
    plt.plot(bins, hist)
    plt.show()
    

    from scipy.optimize import curve_fit
    from scipy.misc import factorial
    
    def poisson(k, lamb):
        return (lamb**k/factorial(k))*np.exp(-lamb)

    def log_factorial(n):
        n = int(n)
        return np.sum(np.log(np.arange(n) + 1))
        
    def log_poisson(x, a, b):
        return a*x + b

    ydata = np.log(hist) + np.asarray([log_factorial(n) for n in bins])
    xdata = bins[~np.isnan(ydata)]
    ydata = ydata[~np.isnan(ydata)]
    xdata = xdata[~np.isinf(ydata)]
    ydata = ydata[~np.isinf(ydata)]
    ydata = ydata[xdata > 0]
    xdata = xdata[xdata > 0]
    plt.plot(xdata, ydata, label='data', linewidth=2)
    
    param, pcov = curve_fit(log_poisson, xdata, ydata)
    a = param[0]
    b = param[1]
    print('lambda = {}'.format(-b))
    plt.plot(xdata, log_poisson(xdata, a, b), 'r', label='log_poisson fit')
    plt.legend(loc=2)
    plt.show()
    

    x_plot = np.arange(max(bins) + 1)
    plt.plot(bins, hist, 'b.')
    plt.plot(x_plot, poisson(x_plot, -b),  'r-')
    plt.show()
