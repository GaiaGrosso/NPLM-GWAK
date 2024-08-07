import numpy as np
from scipy.spatial.distance import pdist

def standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        if std <= 0: continue
        else: dataset_new[:, j] = (dataset[:, j]- mean)*1./ std
    return dataset_new

def inv_standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        dataset_new[:, j] = dataset[:, j] * std + mean
    return dataset_new


def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.                                                                                      
    # use on a (small) sample of reference data (standardize first if necessary)                                                                     
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),1)
