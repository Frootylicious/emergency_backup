import os
import subprocess
import settings.settings as s
import numpy as np
import matplotlib.pyplot as plt

"""
File with commonly used functions.
"""

def read_from_file():
    '''
    Reads all solved network files in the 'nodes_folder' set in 'settings.py'.
    Returns a list of dictionaries with the values from the files in the form:
        [{'c':'u', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00},
            {'c':'c', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00}]
    '''
    filename_list = os.listdir(s.nodes_folder)
    all_combinations = []
    for name in filename_list:
        all_combinations.append({'c':name[0],
                                 'f':name[2],
                                 'a':float(name[5:9]),
                                 'g':float(name[11:15]),
                                 'b':float(name[17:21])})
    return all_combinations


def get_chosen_combinations(**kwargs):
    '''
    Function that extracts the wanted files in the 'self.all_combinations list'.

    To choose all the solved networks in the synchronized flow scheme:
        "self.get_chosen_combinations(f='s')".

    All unconstrained networks with a gamma = 1.00 can be found by:
        self.get_chosen_combinations(c='u', g=1.00)

    returns a list of dictionaries with the desired values.
    For instance:
        [{'c':'u', 'f':'s', 'a':1.00, 'g':1.00, 'b':1.00},
            {'c':'c', 'f':'s', 'a':0.80, 'g':1.00, 'b':0.50}
            ...]
    '''
    def _check_in_dict(dic, kwargs):
        """ Check if values are present in a dictionary.

        Args:
            dic: dictionary to check in.
            kwargs: the keyword arguments to check for in the dictionary.
        Returns:
            boolean: True if all values are present in the dictionary, False if any are not.
        """
        for (name, value) in kwargs.items():
            value = np.array(value)
            if not dic[name] in value:
                return False
        return True

    chosen_combinations = []
    # Checking all combinations.
    for combination in read_from_file():

        if _check_in_dict(combination, kwargs):
            chosen_combinations.append(combination)

    if len(chosen_combinations) == 0:
        # Raise error if no chosen combinations are found.
        raise ValueError('No files with {0} found!'.format(kwargs))
    return chosen_combinations


def quantile(quantile, dataset):
    """
    Docstring for quantile.
    Args:
        quantile [float]: the desired quantile eg. 0.99.
        dataset [list/1-dimensional array]: a timeseries.
    """
    return np.sort(dataset)[int(round(quantile*len(dataset)))]


def quantile_old(quantile, dataset, cutzeros=False):
    """
    Takes a list of numbers, converts it to a list without zeros
    and returns the value of the 99% quantile.
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
        if q > 1 - float(quantile):
            #print 'Found %.3f quantile' % (1 - q)
            return bin_edges[-index]


def storage_size_old(backup_timeseries, q=0.99):
    """
    Docstring
    """
    q = quantile(q, backup_timeseries)
    storage = backup_timeseries - q
    for index, val in enumerate(storage):
        if index == 0:
            if storage[index] < 0:
                storage[index] = 0
        else:
            storage[index] += storage[index - 1]
            if storage[index] < 0:
                storage[index] = 0
    return max(storage)

def storage_size(backup_timeseries, q=0.99):
    """
    """
    q = quantile(0.99, backup_timeseries)
    offset_backup = backup_timeseries - q
    # plt.plot(offset_backup)
    # plt.show()
    storage = np.zeros(len(offset_backup) + 1)
    # i = 0
    for index, val in enumerate(offset_backup):
        # if val > 0:
            # print(i)
            # i += 1
        storage[index] += val
        if storage[index] < 0:
                storage[index] = 0
        storage[index + 1] = storage[index]
        # if val < 0:
            # storage[index] += val
            # if storage[index] < 0:
            #     storage[index] = 0
            # storage[index + 1] = storage[index]


    return (max(storage), storage[:-1], offset_backup)

def storage_size_relative(backup_generation_timeseries, beta_capacity, eta=1):
    '''
    Function that calculates the extreme backup timeseries.

    parameters:
        backup_generation_timeseries: numpy array | a timeseries for the backup generation divided
        by the mean load in that node.

        beta_capacity: number | how much of the backup generation divided by the mean load in that
        node that is not served and thus need extreme backup.

        eta: number | the efficiency of the storage charging and discharging. For instance, eta=0.6
        is a charge efficiency of 0.6 and discharge efficiency of 1/0.6.

    returns:
    '''
    G = np.array(backup_generation_timeseries)
    K = beta_capacity

    # Subtracting the backup generation from the backup capacity.
    K_minus_G = K - G

    # The initial maximum level of the emergency storage.
    S_n_max = 0

    S_n = np.empty_like(K_minus_G)

    for t, K_G in enumerate(K_minus_G):
        eta_loop = np.copy(eta)
        # If we need extreme backup energy.
        if K_G < 0:
            eta_loop **= -1
        # Take care of first timestep.
        if t == 0:
            S_n[t] = np.min((S_n_max, eta_loop * K_G))
        else:
            S_n[t] = np.min((S_n_max, S_n[t - 1] + eta_loop * K_G))

    return(S_n_max - np.min(S_n), S_n)


def get_remote_figures():
    if not os.path.exists(s.remote_figures):
        os.mkdir(s.remote_figures)
    """Copy figures from the result-folder on a remote server"""
    os.system('scp -r {0}. {1}'.format(s.remote_figures_folder, s.remote_figures))
    return


def load_remote_network(filename, N_or_F='N'):
    '''
    Function that copies a given solved network- or flow-object from the remote server
    to a local temp-folder and returns the loaded object.

    parameters:
        filename: string | the name of the object without the suffix and extension:
        'c_s_a0.80_g1.00_b1.00'
        N_or_F:   string | whether you want the network- or flow-object returned.

    returns:
        temp_load: numpy_object | the contents of the chosen file.

    '''

    file_to_check = filename + '_{}.npz'.format(N_or_F)

    print('\nChecking for file {} on server...'.format(file_to_check))
    remote_path = '/home/kofoed/emergency_backup/results/{}/'.format(N_or_F)
    command = 'ssh {0} ls {1}'.format(s.remote_ip, remote_path)
    p = subprocess.Popen(command,
                         universal_newlines=True,
                         shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    files_list = out.split()

    if file_to_check in files_list:
        print('\nFound file on server - copying to temp-directory...')
        filename_total = '{0}:{1}{2}'.format(s.remote_ip, remote_path, file_to_check)
        p2 = subprocess.Popen(['scp', filename_total, 'temp/temp.npz'])
        sts = os.waitpid(p2.pid, 0)

        temp_load = np.load('temp/temp.npz')
        return(temp_load)
    else:
#         print('Did NOT find file')
        raise IOError('No file named {0} on server {1}.'.format(filename, s.remote_ip))


def find_minima(timeseries):
    '''
    Function that takes a timeseries and find the minimum in each connected event (the bottom of the
    dips).

    parameters:
        timeseries: list | a timeseries (here the non-served energy).
    returns: 
        minima: list | a list of the minima in the clustered events.
    '''
    intervals = []
    interval = []
    minima = []

    nonzero = np.nonzero(timeseries)[0]
    diff = np.diff(nonzero)

    for i, d in enumerate(diff):
        interval.append(nonzero[i])
        if d != 1:
            intervals.append(interval)
            interval = []

    if nonzero[-1] - nonzero[-2] == 1:
        interval.append(nonzero[-1])
        intervals.append(interval)
    else:
        intervals.append([nonzero[-1]])

    for interval in intervals:
        minima.append(np.min(timeseries[interval]))

    return(np.array(minima))
