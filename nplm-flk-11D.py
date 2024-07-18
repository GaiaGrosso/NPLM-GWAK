import math, time, datetime, h5py, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from joblib import dump,load
import os as os
from FLKUtils import *
from SampleUtils import *
# shape only
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--toys', type=int, help="toys", required=False, default=100)
parser.add_argument('-w', '--widthperc', type=int, help="k nearest neighbors", required=False, default=90)
parser.add_argument('-s', '--signal', type=str, help="signal name (on the file)", required=True)
parser.add_argument('-r', '--nr', type=int, help="number of reference points", required=True)
parser.add_argument('-d', '--nd', type=int, help="number of data points", required=True)
parser.add_argument('-l', '--lam', type=float, help="falkon lambda", required=False, default=1e-6)
parser.add_argument('-m', '--m', type=int, help="falkon M", required=False, default=1000)

args = parser.parse_args()

M = args.m
lam = args.lam
iterations=1000000000
flk_sigma_perc = args.widthperc        # flk width quantile pair-distance                                                                                          
Nexp = args.toys     # Number of pseudo-experiment
ND = args.nd
NR = args.nr#242*10
w_ref = ND*1./NR
folder_root = "/n/home00/ggrosso/NPLM-GWAK/data/" # where the data are stored
bkg_path = folder_root + 'background_11dim.npy'
data_path = folder_root + args.signal + '_11dim.npy'

# plot options
labels_vars=["H background-like", "L background-like", "H BBH-like",
        "L BBH-like", "H Glitch-like", "L Glitch-like", "H SG lf - like",
        "L SG lf - like", "H SG hf - like", "L SG hf - like",
        "Frequency correlation"]
binsrange = {label : np.linspace(0, 50, 40) for label in labels_vars}
yrange = {label: [0, 5] for label in labels_vars}

# output                                                                                                                                                      
folder_out = '/n/home00/ggrosso/NPLM-GWAK/out/%s/'%(args.signal)
NP = 'M%i_lam%s_iter%i_ND%i_NR%i/'%(M, str(lam), iterations, ND, NR)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)

# Read samples
ref_all = np.load(bkg_path)
data_all = np.load(data_path).reshape((-1, 11))
total_data = data_all.shape[0]*data_all.shape[1]

mean_all, std_all = np.mean(ref_all, axis=0), np.std(ref_all, axis=0)
ref_all_std  = standardize(ref_all, mean_all, std_all).astype('f')
data_all_std = standardize(data_all, mean_all, std_all).astype('f')

# candidate sigma                                                                                                                                           
flk_sigma = candidate_sigma(ref_all_std[:1000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

if args.signal=='background':
    # calibration
    print('Calibration')
    random_seeds=np.random.randint(0, 100000, Nexp)
    tnull = np.array([])
    if os.path.exists('%s/tnull_sigma%s.npy'%((folder_out+NP), flk_sigma)):
        tnull = np.load('%s/tnull_sigma%s.npy'%((folder_out+NP), flk_sigma))
        print(tnull.shape[0],' previous experiments. Collecting them.')
    for i in np.arange(Nexp):
        seed = int(random_seeds[i])
        rng = np.random.default_rng(seed=seed)
        index = np.arange(ND+NR)
        rng.shuffle(index)
        input_tmp = ref_all_std[index].astype('f')
        label_R = np.zeros((NR,))
        label_D = np.ones((ND,))
        labels  = np.concatenate((label_D,label_R), axis=0).reshape((-1, 1)).astype('f')
        plot_reco=False
        verbose=False

        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
        t_tmp, pred_tmp = run_toy('toy%i'%(i), input_tmp, labels, w_ref, flk_config, seed,
                                  plot=plot_reco, verbose=verbose, savefig=plot_reco, output_path='./')
        tnull = np.append(tnull, t_tmp)
    np.save('%s/tnull_sigma%s.npy'%(folder_out+NP, flk_sigma), tnull)
else:
    # t obs
    if not os.path.exists('%s/tnull_sigma%s.npy'%((folder_out+NP).replace(args.signal, 'background'), flk_sigma)):
        print('%s/tnull_sigma%s.npy'%((folder_out+NP).replace(args.signal, 'background'), flk_sigma), 'does not exists.')
        print('You need to compute tnull first. Exit.')
        exit()
    tnull = np.load('%s/tnull_sigma%s.npy'%((folder_out+NP).replace(args.signal, 'background'), flk_sigma))
    # scan windows
    counter=0
    random_seeds=np.random.randint(0, 100000, int(total_data/ND))
    pval_list = np.array([])
    tobs_list = np.array([])
    while ((counter+1)*ND)<total_data:
        seed = int(random_seeds[counter])
        rng = np.random.default_rng(seed=seed)
        index_ref=np.arange(ref_all_std.shape[0])
        rng.shuffle(index_ref)
        ref_sw = ref_all_std[index_ref[:NR]]
        data_sw = data_all_std[counter*ND:(counter+1)*ND, :]
        input_sw = np.concatenate((ref_sw, data_sw), axis=0).astype('f')
        label_R = np.zeros((NR,))
        label_D = np.ones((ND,))
        labels  = np.concatenate((label_D,label_R), axis=0).reshape((-1, 1)).astype('f')
        # tobs
        plot_reco = False
        verbose = False
        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
        tobs, pred_obs = run_toy('tobs%i_sig%s'%(counter,flk_sigma), input_sw, labels, w_ref, flk_config, seed,
                             plot=plot_reco, verbose=verbose, savefig=plot_reco, output_path=folder_out+NP,
                                 #binsrange=binsrange,
                                 yrange=yrange,
                                 xlabels=labels_vars,
        )
        # pval
        pval = np.sum(tnull>=tobs)*1./len(tnull)
        pval_list = np.append(pval_list, pval)
        tobs_list = np.append(tobs_list, tobs)
        print('tobs: ', tobs)
        print('p-val: ', pval)
        if pval<0.01:
            plot_reconstruction(data=input_sw[labels.flatten()==1], weight_data=1,
                                ref=input_sw[labels.flatten()==0], weight_ref=w_ref,
                                ref_preds=pred_obs[labels.flatten()==0],
                                yrange=yrange,#binsrange=binsrange,
                                xlabels=labels_vars,
                                save=True, save_path=folder_out+NP+'/plots/',
                                file_name='tobs%i_sig%s'%(counter,flk_sigma)+'.pdf'
                )
        counter+=1
        if counter==Nexp:break
    # save stuff
    print('save stuff')
    np.save('%s/tobs_sigma%s.npy'%(folder_out+NP, flk_sigma), tobs_list)
    np.save('%s/pval_sigma%s.npy'%(folder_out+NP, flk_sigma), pval_list)

