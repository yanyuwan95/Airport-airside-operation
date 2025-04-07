import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
#pd.set_option('max_rows', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_file_name(folder_name):
    files_location = os.path.join(os.getcwd(), folder_name)
    lst_file_name = []

    for file_name in os.listdir(files_location):
        file = os.path.join(files_location, file_name)
        lst_file_name.append(file)

    return lst_file_name

def loadConfigForAirport(airport):
    ### load configuration
    config = ConfigParser()
    config.read(airport + '_beginLoadParameter.ini')
    return config

def color_dict(lst_cluster_name, cmapname):
    cmap = plt.get_cmap(cmapname)
    colors = cmap(np.linspace(0.2, 1, len(lst_cluster_name)))
    dict_colors = dict(zip(lst_cluster_name, colors))
    return dict_colors

def split_list(x, pre_list):
    return [pre_list[i:i + x] for i in range(0, len(pre_list), x)]

def find_best_dist(data, lower, upper):

    size = len(data)  # number of samples to generate

    x = np.linspace(lower, upper, len(data))

    # Fit a range of distributions to the data
    loc, scale = stats.uniform.fit(data)
    mu, std = stats.norm.fit(data)
    lower, locg, scaleg = stats.gamma.fit(data)
    loce, scalee = stats.expon.fit(data)
    s, locl, scalel = stats.lognorm.fit(data)

    uniform_pdf = stats.uniform.pdf(x, loc=loc, scale=scale)
    norm_pdf = stats.norm.pdf(x, loc=mu, scale=std)
    gamma_pdf = stats.gamma.pdf(x, lower, locg, scaleg)
    expon_pdf = stats.expon.pdf(x, loce, scalee)
    lognorm_pdf = stats.lognorm.pdf(x, s, locl, scalel)

    lst_pdf = [uniform_pdf, norm_pdf, gamma_pdf, expon_pdf, lognorm_pdf]

    # Evaluate goodness-of-fit using the Kolmogorov-Smirnov test
    dist_names = ['uniform', 'norm', 'gamma', 'expon', 'lognorm']
    dist_params = [(loc, scale), (mu, std), (lower, locg, scaleg), (loce, scalee), (s, locl, scalel)]

    dict_pdf_with_name = dict(zip(dist_names, lst_pdf))
    dict_colors = color_dict(dist_names, cmapname='Paired')
    print(dist_params)

    dict_result = {}
    for i, dist_name in enumerate(dist_names):
        dist = getattr(stats, dist_name)
        p_value = stats.kstest(data, dist_name, args=dist_params[i])[1]
        dict_result[dist_name] = '{:.3f}'.format(p_value)
        print("{:12s} p-value = {:.3f}".format(dist_name, p_value))

    # Find the best-fitting distribution
    best_fit_name = dist_names[
        np.argmax([stats.kstest(data, dist_name, args=dist_params[i])[1] for i, dist_name in enumerate(dist_names)])]
    best_fit_params = dist_params[
        np.argmax([stats.kstest(data, dist_name, args=dist_params[i])[1] for i, dist_name in enumerate(dist_names)])]
    new_df = pd.DataFrame({'best_fit_name': [best_fit_name],
                           'best_fit_params': [best_fit_params],
                           'dict_fit_parms': [dict_result]})
    print("Best fit: {} with parameters {}".format(best_fit_name, best_fit_params))

    fig, (ax1,ax2) = plt.subplots(1,2)

    ax1.scatter(range(len(data)), data)
    for k, v in dict_pdf_with_name.items():
        ax2.plot(v, color=dict_colors[k],label=k)
    plt.legend()

    plt.show()
    return new_df