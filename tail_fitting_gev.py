#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import toolbox.plotting as tbplt
import settings.tools as t
from statistics import stdev, mean
import seaborn
import settings.settings as s
seaborn.set_style('whitegrid')
# plt.style.use('fivethirtyeight')  


def get_timeseries(country, alpha, gamma):
    countries = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG',
                'GR', 'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE',
                'SE', 'DK', 'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']
    country_dict = dict(zip(countries, list(range(len(countries)))))
    balancing = np.sum(N.f.balancing, axis=0)
    return balancing
#     return np.load('balancing/%.2f_%.2f.npz'%(alpha, gamma))['arr_0'][country_dict[country]]

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

def normal_distribution(x, mu, sigma):
    return 1./np.sqrt(2.*np.pi*sigma)*np.exp(-(x - mu)**2/(2*sigma**2))


def fit_gev_to_histogram(timeseries, q=0.90):

    cum_dens,  bin_edges = cdf(timeseries, normed=False)
    cum_dens /= float(max(cum_dens))

    bins = bin_edges[:-1] + 0.5*np.diff(bin_edges)
#     bins = bins/1000.0

    # -- Getting only the q'th quantile and up --
    bins = bins[cum_dens > q]
    cum_dens = cum_dens[cum_dens > q]

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


    # Doing a linear regression on xdata vs ydata
    popt, pcov = curve_fit(lin, xdata, ydata)
    a = popt[0]
    b = popt[1]

    # Plotting the found regression to the data
    # x0 = xdata[0]
    # x1 = xdata[-1]
    # y0 = lin(x0, a, b)
    # y1 = lin(x1, a, b)

    # Getting the values from the a and b values
    xi = 1.0/a
    sigma2 = -np.exp(xi*b)
    mu2 = max(bins)*1.001
    sigma = xi/sigma2
    mu = mu2 + sigma/xi

    params = (mu, sigma, xi)
    params2 = (mu2, sigma2, xi)
    return params2
    
def GEV_dens(x, mu, sigma, xi):
    return 1.0/sigma*(1 + xi*(x - mu)/sigma)**(-1.0/xi - 1)*np.exp(-(1 + xi*(x - mu)/sigma)**(-1.0/xi))

def normal_distribution(x, mu, sigma):
    return 1./np.sqrt(2.*np.pi*sigma)*np.exp(-(x - mu)**2/(2*sigma**2))

if __name__ == '__main__':
    alpha = 0.80
#     DE_full_flow = get_timeseries('DE', alpha=alpha, gamma=1.00)
    N_inf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=0.80, b=np.inf, g=1.00))
    N_0 = np.load(s.nodes_fullname.format(c='c', f='s', a=0.80, b=0, g=1.00))
    DE_full_flow = N_inf.f.balancing[18]
    DE_no_flow = N_0.f.balancing[18]

#     load = np.load('/home/simon/Dropbox/Root/Data/ISET/ISET_country_DE.npz')['L']
#     load *= 1000
#     load = np.sum(N_inf.f.load, axis=0)
    load = N_inf.f.load[18]

    # Finding fitting parameters
    DE_full_flow /= np.mean(load)
    DE_no_flow /= np.mean(load)
    params_full_flow = fit_gev_to_histogram(DE_full_flow)
    params_no_flow = fit_gev_to_histogram(DE_no_flow)

    full_sigma = stdev(DE_full_flow[DE_full_flow>=1e-8]/1000.)
    full_mean = mean(DE_full_flow[DE_full_flow>=1e-8]/1000.)
    no_sigma = stdev(DE_no_flow/1000.)
    no_mean = mean(DE_no_flow/1000.)

    
    # Setting up for the plotting
    fhist_full_flow, bin_edges_full_flow = t.fancy_histogram(DE_full_flow,
                                                                 bins=100,
                                                                 density=True)
    fhist_no_flow, bin_edges_no_flow = t.fancy_histogram(DE_no_flow,
                                                                 bins=100,
                                                                 density=True)
    fhist_full_flow = fhist_full_flow[2:] # to remove the spike at 0
    bin_edges_full_flow = bin_edges_full_flow[2:]
    bin_edges_full_flow /= 1000.0 # because of weird unit convention in the data
    fhist_no_flow = fhist_no_flow[2:] # to remove the spike at 0
    bin_edges_no_flow = bin_edges_no_flow[2:]
    bin_edges_no_flow /= 1000.0 # because of weird unit convention in the data


    # Scaling the plots properly
    gevs_full_flow = GEV_dens(bin_edges_full_flow, *params_full_flow)
    normal_full_flow = normal_distribution(bin_edges_full_flow,
                                           full_mean, full_sigma)
    gevs_full_flow /= np.sum(gevs_full_flow)
    normal_full_flow /= np.sum(normal_full_flow)
    fhist_full_flow /= np.sum(fhist_full_flow)
    gevs_no_flow = GEV_dens(bin_edges_no_flow, *params_no_flow)
    normal_no_flow = normal_distribution(bin_edges_no_flow,
                                         no_mean, no_sigma)
    gevs_no_flow /= np.sum(gevs_no_flow)
    normal_no_flow /= np.sum(normal_no_flow)
    fhist_no_flow /= np.sum(fhist_no_flow)

    
    # Only plotting gev for the 90% quantile and up. 
    fhist_full_cumsum = np.cumsum(fhist_full_flow)
    fhist_no_cumsum = np.cumsum(fhist_no_flow)
    fhist_full_90 = fhist_full_flow[fhist_full_cumsum > 0.90*max(fhist_full_cumsum)]
    fhist_no_90 = fhist_no_flow[fhist_no_cumsum > 0.90*max(fhist_no_cumsum)]
    gevs_full_90 = gevs_full_flow[fhist_full_cumsum > 0.90*max(fhist_full_cumsum)]
    gevs_no_90 = gevs_no_flow[fhist_no_cumsum > 0.90*max(fhist_no_cumsum)]
    normal_full_90 = normal_full_flow[fhist_full_cumsum > 0.90*max(fhist_full_cumsum)]
    normal_no_90 = normal_no_flow[fhist_no_cumsum > 0.90*max(fhist_no_cumsum)]
    bin_full_90 = bin_edges_full_flow[fhist_full_cumsum > 0.90*max(fhist_full_cumsum)]
    bin_no_90 = bin_edges_no_flow[fhist_no_cumsum > 0.90*max(fhist_no_cumsum)]

    plt.plot(1000*bin_edges_full_flow, fhist_full_flow,
             linewidth=2,
             label=r'$\beta=\infty$')
    # plt.plot(1000*bin_edges_full_flow, normal_full_flow,
    #          linewidth=2,
    #          label=r'Normal')
    plt.plot(1000*bin_edges_no_flow, fhist_no_flow,
             linewidth=2,
             label=r'$\beta=0$')
    plt.plot(1000*bin_full_90, gevs_full_90, '--',
             linewidth=2,
             label='GEV fit')
    plt.plot(1000*bin_no_90, gevs_no_90, '--',
             linewidth=2,
             label='GEV fit')
    plt.ylabel(r'$p(G_n^B)$', fontsize=18)
    plt.xlabel(r'$G_n^B/\left\langle L_n \right\rangle$', fontsize=18)
    plt.legend(prop={'size':15})
    plt.tight_layout()
#     plt.savefig(r'/home/simon/Dropbox/Speciale/LaTeX/Graphics/full_no_flow_gev_fit.pdf')
    plt.savefig(s.results_folder + 'full_no_flow_gev_fit.pdf')
    plt.clf()
    plt.close()
    # plt.show()

    plt.plot(1000*bin_full_90, np.log(fhist_full_90),
             linewidth=2,
             label=r'$\beta=\infty$')
    plt.plot(1000*bin_no_90, np.log(fhist_no_90),
             linewidth=2,
             label=r'$\beta=0$')
    plt.plot(1000*bin_full_90, np.log(gevs_full_90), '--',
             linewidth=2,
             label='GEV fit')
    plt.plot(1000*bin_no_90, np.log(gevs_no_90), '--',
             linewidth=2,
             label='GEV fit')
    plt.plot(1000*bin_full_90, np.log(normal_full_90), '.',
             linewidth=1, markersize=5,
             label='Normal fit')
    plt.plot(1000*bin_no_90, np.log(normal_no_90), '.',
             linewidth=1, markersize=5,
             label='Normal fit')
    plt.ylabel(r'$\log\ p(G_n^B)$', fontsize=18)
    plt.xlabel(r'$G_n^B/\left\langle L_n \right\rangle$', fontsize=18)
    plt.legend(loc=3, prop={'size':15})
    plt.tight_layout()
#     plt.savefig(r'/home/simon/Dropbox/Speciale/LaTeX/Graphics/log_full_no_flow_gev_fit.pdf')
    plt.savefig(s.results_folder + 'log_full_zero_flow_gev_fit.pdf')
    plt.clf()
    plt.close()
    # plt.show()

def make_table_parameters():
    values = np.linspace(0, 1, 11)
    m2_inf = np.empty_like(values)
    s2_inf = np.empty_like(values)
    x_inf = np.empty_like(values)
    m2_0 = np.empty_like(values)
    s2_0 = np.empty_like(values)
    x_0 = np.empty_like(values)
    for i, n in enumerate(values):
        N_inf = np.load(s.nodes_fullname_inf.format(c='c', f='s', a=n, g=1.00, b=np.inf))
        N_0 = np.load(s.nodes_fullname.format(c='c', f='s', a=n, g=1.00, b=0))
        L = np.mean(N_inf.f.load[18])
        B_inf = N_inf.f.balancing[18]
        B_0 = N_0.f.balancing[18]

        (m2_0[i], s2_0[i], x_0[i]) = fit_gev_to_histogram(B_0/L)
        (m2_inf[i], s2_inf[i], x_inf[i]) = fit_gev_to_histogram(B_inf/L)
    
#     l1 = ['{:.1f}'.format(a) for a in values]
    a = values
    results = np.array((m2_0, s2_0, x_0, m2_inf, s2_inf, x_inf)).T
    print(results)
    np.savetxt(s.results_folder + 'gev_table.txt', results,
            delimiter=',',
            fmt='%.5f')






