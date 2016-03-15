#! /usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def get_load(country):
    return np.load('/home/simon/Dropbox/Root/Data/ISET/ISET_country_%s.npz' %(country))['L']

def get_timeseries(country):
    countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                 'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                 'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
    country_dict = dict(zip(countries, list(range(len(countries)))))
    return np.load('balancing/0.90_1.00.npz')['arr_0'][country_dict[country]]

def lin(x, a, b):
    return a*x + b

def cdf(timeseries, normed=True):
    timeseries = timeseries[timeseries > 1e-6]
    hist, bin_edges = np.histogram(timeseries, bins=10000)
    d = np.diff(bin_edges)[0]
    if normed:
        c = np.cumsum(hist)
        return c/float(max(c)), bin_edges
    else:
        return np.cumsum(hist*d), bin_edges

def inv_cdf(timeseries, quantile):
    c, bin_edges = cdf(timeseries)
    bins = bin_edges[:-1] + 0.5*np.diff(bin_edges)
    return bins[c > quantile][0]

country = 'DE'
DK = get_timeseries(country)
cum_dens, bin_edges = cdf(DK, normed=False)
cum_df = cum_dens
cum_dens = cum_dens/float(max(cum_dens))
bins = bin_edges[:-1] + 0.5*np.diff(bin_edges)
bins = bins/1000.0

# -- Getting only the 90% quantile and up --
bins = bins[cum_dens > 0.90]
cum_df = cum_df[cum_dens > 0.90]
cum_dens = cum_dens[cum_dens > 0.90]


# As x vs. y, we want to plot log(bins) vs. -log(-log(cum_dens))
xdata = np.log(-bins + max(bins))
ydata = -np.log(-np.log(cum_dens))
ydata = ydata[:len(ydata)/2.0]
xdata = xdata[:len(xdata)/2.0]

# For ydata, there is a chance that we will get som nan or inf values,
# so we remove those after the fact.
print('Removing nan\'s and inf\'s')
xdata = xdata[~np.isnan(ydata)]
xdata = xdata[~np.isinf(ydata)]
ydata = ydata[~np.isnan(ydata)]
ydata = ydata[~np.isinf(ydata)]

#plt.plot(xdata, ydata)
#plt.show()

# Doing a linear regression on xdata vs ydata
popt, pcov = curve_fit(lin, xdata, ydata)
a = popt[0]
b = popt[1]

# Plotting the found regression to the data
x0 = xdata[0]
x1 = xdata[-1]
y0 = lin(x0, a, b)
y1 = lin(x1, a, b)
# reg = plt.plot([x0, x1], [y0, y1], 'y', label='Linear regression')
# data = plt.plot(xdata, ydata, 'bo', label='Data')
# plt.xlabel(r'$\log (-x)$', fontsize=18)
# plt.ylabel(r'$-\log(-\log ($CDF$))$', fontsize=18)
# plt.legend(loc=2)
# plt.show()

# Getting the values from the a and b values
xi = 1.0/a
sigma2 = -np.exp(xi*b)
mu2 = max(bins)
sigma = xi/sigma2
mu = mu2 + sigma/xi

def GEV_cum_dens(x, mu, sigma, xi):
    return np.exp(-(1 + xi*(x - mu)/sigma)**(-1.0/xi))

gevs = [GEV_cum_dens(x, mu, sigma, xi) for x in bins]
# plt.plot(bins, gevs, 'y')
# plt.plot(bins, cum_dens, 'b.')
# plt.ylabel(r'Quantile', fontsize=18)
# plt.xlabel(r'Backup [GWh]', fontsize=18)
# plt.show()


#------------------------------------------
fig = plt.figure(figsize=(16,10))
ax1 = fig.add_subplot(211)
ax1.plot(gevs, cum_dens, 'b.', markersize=8, label='QQ')
orange = '#FFC400'

ax1.plot([0.9, 1.0], [0.9, 1.0], color=orange, linewidth=2.0)
#plt.axis([min(0.9, min(gevs)), max(1, max(gevs)), min(0.9, min(cum_dens)), max(1, max(cum_dens))])
ax1.set_xlabel(r'GEV CDF', fontsize=18)
ax1.set_ylabel(r'Observed CDF', fontsize=18)
ax1.set_title(r'QQ backup for %s, $\alpha = 0.80$, $\gamma = 2.00$' %(country), fontsize=18)
ax1.grid()
ax1.legend(loc=2)

new_y = ax1.twinx()
rounded_CDF = np.around(cum_dens, decimals=3)
new_ticks = [bins[rounded_CDF == x][-1] for x in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]]
new_ticks = ['%.2f' % x for x in new_ticks]
#new_ticks = ['%.2f' % x for x in bins[cum_dens == 0.]]
new_y.set_ylim(ax1.get_ylim())
new_y.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
new_y.set_yticklabels(new_ticks)
new_y.set_ylabel(r'Backup [GWh]', fontsize=18)
# #fig.savefig('test.pdf', bbox_tight=True)

ax2 = fig.add_subplot(212)
#print(gevs)
def full_inverse_GEV(q, mu2, sigma2, xi):
    return ((-np.log(q))**(-xi))/sigma2 + mu2
q_bins = np.asarray([full_inverse_GEV(q, mu2, sigma2, xi) for q in cum_dens])
ax2.plot(q_bins, bins, 'b.', markersize=8)
ax2.plot([min(q_bins), max(q_bins)], [min(q_bins), max(q_bins)], color=orange, linewidth=2.0)
ax2.set_xlabel(r'Backup quantile [GWh]', fontsize=18)
ax2.set_ylabel(r'Backup quantile [GWh]', fontsize=18)
ax2.grid()
fig.savefig('test.pdf', bbox_tight=True)

#many_gevs = np.asarray([GEV_cum_dens(x, mu, sigma, xi) for x in np.linspace(min(bins)*0.9, max(bins), 1e6)])


