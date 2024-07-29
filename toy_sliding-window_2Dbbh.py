import glob, h5py, math, time, os, json, random, yaml, argparse, datetime
from scipy.stats import norm, expon, chi2, uniform, chisquare
from pathlib import Path
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

from NNUtils import *
from PlotUtils import *
from SampleUtils import *
parser   = argparse.ArgumentParser()

parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-s', '--slidingwindow', type=int, help="index at which to start reading the data", required=True)
args     = parser.parse_args()

# random seed
seed = args.slidingwindow+1
if seed==None:
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('Random seed:'+str(seed))

# setup parameters
with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)

# problem definition
NR = config_json["N_Ref"]
ND = config_json["N_Data"]
signal = config_json["signal"]
w_ref = ND*1./NR
sliding_window=args.slidingwindow

# plot options                                                                                  
labels_vars=["H background-like", "L background-like", "H BBH-like", "L BBH-like", "H Glitch-like", "L Glitch-like", "H SG lf - like",  "L SG lf - like", "H SG hf - like", "L SG hf - like", "Frequency correlation"]
binsrange = {label : np.linspace(0, 50, 40) for label in labels_vars}
yrange = {label: [-5, 5] for label in labels_vars}

# data     
folder_root = "/n/home00/ggrosso/NPLM-GWAK/data/" # where the data are stored
bkg_path = folder_root + 'background_11dim.npy'
data_path = folder_root +'/signals/'+ signal + '_11dim.npy'
ref_all = np.load(bkg_path)[:, 2:4]
data_all = np.load(data_path).reshape((-1, 11))
data_all = data_all[sliding_window*ND:(sliding_window+1)*ND, 2:4]

mean_all, std_all = np.mean(ref_all, axis=0), np.std(ref_all, axis=0)
mean_R, std_R = mean_all, std_all
ref_all_std  = standardize(ref_all, mean_all, std_all).astype('f')
featureData = standardize(data_all, mean_all, std_all).astype('f')
featureRef = ref_all_std[:NR, :]
#featureData = data_all_std[sliding_window*ND:(sliding_window+1)*ND, :]
print(len(featureData))
# labels
label_R = np.zeros((NR, 1))
label_D = np.ones((ND, 1))
weights_D = np.ones((ND, 1))
weights_R = np.ones((NR, 1))*w_ref
# apply selection and std                                                                                          
target  = np.concatenate((label_D, label_R), axis=0)
weights = np.concatenate((weights_D, weights_R), axis=0)
target  = np.concatenate((target, weights), axis=1)
feature = np.concatenate((featureData, featureRef), axis=0)

# model definition
M = config_json["number_centroids"]
d = 2
resolution_scale = np.array(config_json["resolution_scale"]).reshape((-1,))
resolution_const = np.array(config_json["resolution_const"]).reshape((-1,))
print('resolution constants')
print(resolution_const)

# initialization
## initialize M to match the data.
## if not enough data random pick from ref extra points.
if M<=ND:
    idx = np.random.randint(len(featureData), size=M)
    centroids_init = featureData[idx, :]
else:
    idx = np.random.randint(len(featureRef), size=M-ND)
    centroids_init = np.concatenate((featureData, featureRef[idx, :]), axis=0)
    
widths_init    = np.ones((M, d))*config_json["width_init"]                         
coeffs_init    = np.random.uniform(low=-100, high=100, size=(M, 1))

lam_coeffs   = config_json["coeffs_reg_lambda"]
lam_widths   = config_json["widths_reg_lambda"]
lam_entropy  = config_json["entropy_reg_lambda"]

patience     = config_json["patience"]
plt_patience = config_json["plt_patience"]
total_epochs = config_json["epochs"]
sub_epochs   = config_json["sub_epochs"]
coeffs_clip  = config_json["coeffs_clip"]

train_coeffs   = config_json["train_coeffs"]
train_widths   = config_json["train_widths"]
train_centroids= config_json["train_centroids"]

# convert to tf tensors                                       
feature = tf.convert_to_tensor(feature, dtype=tf.float64)
target  = tf.convert_to_tensor(target, dtype=tf.float64)

##### define output path ######################
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/s%i/'%(seed)
folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
output_folder = folder_out

# run     
model = KernelMethod(input_shape=(None, 1),
                     centroids=centroids_init,
                     widths=widths_init,
                     coeffs=coeffs_init,
                     resolution_const=resolution_const,
                     resolution_scale=resolution_scale,
                     coeffs_clip = coeffs_clip,
                     train_widths=train_widths,
                     train_coeffs=train_coeffs,
                     train_centroids=train_centroids,
                     positive_coeffs=False,
                    )

pred = model.call(feature)
# init loss
print('initial loss:')  
loss_value = 0
nplm_loss_value = NPLMLoss(target, pred)
print("nplm ", nplm_loss_value)
print("coeff", L2Regularizer(model.get_coeffs()))
print("entropy", CentroidsEntropyRegularizer(model.get_centroids_entropy()))

loss_value += nplm_loss_value
loss_value += lam_coeffs * L2Regularizer(model.get_coeffs())
loss_value += lam_widths * L2Regularizer(model.get_widths())
loss_value += lam_entropy * CentroidsEntropyRegularizer(model.get_centroids_entropy())

# init history vectors
widths_history    = model.get_widths().numpy().reshape((1, M, -1))
centroids_history = model.get_centroids().numpy().reshape((1, M, -1))
coeffs_history    = model.get_coeffs().numpy().reshape((1, M))
nplm_loss_history = np.array([nplm_loss_value])
epochs_history    = np.array([0])
loss_history      = np.array([loss_value])

t1 =time.time()

#print(model.trainable_variables)
#print(model.kernel_layer.trainable_variables)
#print(model.coeffs)
#print(model.kernel_layer.widths)
#print(model.kernel_layer.centroids)
#exit()

# define alternate training
alternate_training={}
if train_coeffs:
    alternate_training["train_coeffs"] = {}
    alternate_training["train_coeffs"]["train_vars"]=model.trainable_variables[0:1]
    alternate_training["train_coeffs"]["reg_coeffs"]=lam_coeffs
    alternate_training["train_coeffs"]["reg_widths"]=0
    alternate_training["train_coeffs"]["reg_centr"]=0
    alternate_training["train_coeffs"]['opt'] = Adam(learning_rate=1e-3)

if train_widths:
    alternate_training["train_widths"] = {}
    alternate_training["train_widths"]["train_vars"]=model.trainable_variables[0+1*(train_centroids==True)+1*(train_widths==True):1+1*(train_centroids==True)+1*(train_widths==True)]
    alternate_training["train_widths"]["reg_coeffs"]=0
    alternate_training["train_widths"]["reg_widths"]=lam_widths
    alternate_training["train_widths"]["reg_centr"]=lam_entropy
    alternate_training["train_widths"]['opt'] = Adam(learning_rate=1e-3)
    
if train_centroids:
    alternate_training["train_centroids"] = {}
    alternate_training["train_centroids"]["train_vars"]=model.trainable_variables[0+1*(train_centroids==True):1+1*(train_centroids==True)]
    alternate_training["train_centroids"]["reg_coeffs"]=0
    alternate_training["train_centroids"]["reg_widths"]=0
    alternate_training["train_centroids"]["reg_centr"]=lam_entropy
    alternate_training["train_centroids"]['opt'] = Adam(learning_rate=1e-3)
    
# training                                                                                                                        
for i in range(int(total_epochs)):
    for train_step in list(alternate_training.keys()):
        train_vars = alternate_training[train_step]["train_vars"]
        optimizer = alternate_training[train_step]['opt']
        for j in range(sub_epochs):
            with tf.GradientTape() as tape:
                reg_coeffs, reg_widths, reg_entr = 0, 0, 0
                pred = model.call(feature)
                nplm_loss_value = NPLMLoss(target, pred)
                loss_value = nplm_loss_value                                                      
                if alternate_training[train_step]["reg_coeffs"]:
                    l_coeffs = alternate_training[train_step]["reg_coeffs"]
                    reg_coeffs = l_coeffs * L2Regularizer(model.get_coeffs())
                    loss_value += reg_coeffs
                if alternate_training[train_step]["reg_widths"]:
                    l_widths = alternate_training[train_step]["reg_widths"]
                    reg_widths = l_widths * L2Regularizer(model.get_widths())
                    loss_value += reg_widths
                if alternate_training[train_step]["reg_centr"]:
                    l_entr = alternate_training[train_step]["reg_centr"]
                    reg_entr = l_entr * CentroidsEntropyRegularizer(model.get_centroids_entropy())
                    loss_value += reg_entr
            grads = tape.gradient(loss_value, train_vars)
            optimizer.apply_gradients(grads,train_vars)
            model.clip_coeffs()
        if not (i%patience):
            widths_history    = np.concatenate((widths_history, model.get_widths().numpy().reshape((1, M, -1))), axis=0)
            centroids_history = np.concatenate((centroids_history, model.get_centroids().numpy().reshape((1, M, -1))), axis=0)
            coeffs_history    = np.concatenate((coeffs_history, model.get_coeffs().numpy().reshape(1, M)), axis=0)
            nplm_loss_history = np.append(nplm_loss_history, np.array([nplm_loss_value]))
            loss_history   = np.append(loss_history, loss_value)
            epochs_history = np.append(epochs_history, i+1)
            print('epoch: %i, loss: %f, NPLM_x: %f, COEFFS: %f, WIDTHS: %f, ENTROPY: %f'%(int(i+1), loss_value,nplm_loss_value, reg_coeffs, reg_widths, reg_entr))
            coeffs_final    = coeffs_history[-1, :]
            widths_final    = widths_history[-1, :, :]
            centroids_final = centroids_history[-1, :, :]
            
        if np.isnan(loss_value): break
    if ((i%plt_patience) or (i==0)) and (i!=(total_epochs-1)): continue # plot
    if np.isnan(loss_value):break
    fig = plt.figure(figsize=(9,6))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
    plt.plot(epochs_history[2:], loss_history[2:], label='loss')
    font=font_manager.FontProperties(family='serif', size=18)
    plt.legend(prop=font, loc='best')
    plt.ylabel('Loss', fontsize=18, fontname='serif')
    plt.xlabel('Epochs', fontsize=18, fontname='serif')
    plt.xticks(fontsize=16, fontname='serif')
    plt.yticks(fontsize=16, fontname='serif')
    plt.grid()
    plt.savefig(output_folder+'loss.pdf')
    plt.close()

    fig = plt.figure(figsize=(9,6))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
    plt.plot(epochs_history, nplm_loss_history, label='NPLM loss')
    font=font_manager.FontProperties(family='serif', size=18)
    plt.legend(prop=font, loc='best')
    plt.ylabel('Loss', fontsize=18, fontname='serif')
    plt.xlabel('Epochs', fontsize=18, fontname='serif')
    plt.xticks(fontsize=16, fontname='serif')
    plt.yticks(fontsize=16, fontname='serif')
    plt.grid()
    plt.savefig(output_folder+'NPLMloss.pdf')
    plt.close()
    if train_centroids:
        for k in range(d):
            fig = plt.figure(figsize=(9,6))
            fig.patch.set_facecolor('white')
            ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
            for m in range(M):
                plt.plot(epochs_history, inv_standardize(centroids_history[:, m, k:k+1],
                                                                mean_R[k:k+1], std_R[k:k+1]), label='%i'%(m))
            font=font_manager.FontProperties(family='serif', size=14)
            plt.ylabel('Centroid loc', fontsize=18, fontname='serif')
            plt.xlabel('Epochs', fontsize=18, fontname='serif')
            plt.xticks(fontsize=16, fontname='serif')
            plt.yticks(fontsize=16, fontname='serif')
            plt.grid()
            plt.savefig(output_folder+'centroids_%i.pdf'%(k))
            plt.close()
    
    fig = plt.figure(figsize=(9,6))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
    for m in range(M):
        plt.plot(epochs_history, coeffs_history[:, m], label='%i'%(m))
    font=font_manager.FontProperties(family='serif', size=14)
    plt.ylabel('Coeffs', fontsize=18, fontname='serif')
    plt.xlabel('Epochs', fontsize=18, fontname='serif')
    plt.xticks(fontsize=16, fontname='serif')
    plt.yticks(fontsize=16, fontname='serif')
    plt.grid()
    plt.savefig(output_folder+'coeffs.pdf')
    plt.close()

    fig = plt.figure(figsize=(9,6))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
    pred_norm = tf.math.sigmoid(pred[:, 0])
    w_norm = target[:, 1]
    bins = np.linspace(0., 1, 20)
    hD = plt.hist(pred_norm[target[:, 0]==1],weights=w_norm[target[:, 0]==1], bins=bins,
                  label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
    hR = plt.hist(pred_norm[target[:, 0]==0], weights=w_norm[target[:, 0]==0],
                  color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
    plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
    font = font_manager.FontProperties(family='serif', size=16)
    l    = plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
    font = font_manager.FontProperties(family='serif', size=18)
    plt.yticks(fontsize=16, fontname='serif')
    plt.xticks(fontsize=16, fontname='serif')
    plt.xlim(0, 1)
    plt.ylabel("events", fontsize=22, fontname='serif')
    plt.xlabel("classifier output", fontsize=22, fontname='serif')
    plt.yscale('log')
    plt.savefig(output_folder+'out.pdf')
    plt.close()
    ####
    
    w_dat = target[:, 1]
    ref_preds = model.call(feature[target[:, 0]==0])
    dat = inv_standardize(feature, mean_R, std_R)
    centroids_m_final = inv_standardize(centroids_history[-1, :, :], mean_R, std_R)
    for k in range(d):
        centroids_m_final_k = centroids_m_final[:, k:k+1]
        dat_k = dat[:, k:k+1]
        plot_reconstruction(data=dat_k[target[:, 0]==1],
                            weight_data=w_dat[target[:, 0]==1],
                            ref=dat_k[target[:, 0]==0],
                            weight_ref=w_dat[target[:, 0]==0],
                            ref_preds=[ref_preds],
                            ref_preds_labels=['NPLM'],
                            centroids=centroids_m_final_k,
                            t_obs=None, df=None,
                            file_name='reco_%i_epoch%i.pdf'%(k, i), save=True,
                            save_path=output_folder,
                            xlabels=[labels_vars[k]], yrange=yrange,
                            bins=np.linspace(np.min(dat_k), np.max(dat_k), 30))


t2=time.time()
print('End training')
print('execution time: ', t2-t1)
pred = model.call(feature)
nplm_loss_final = NPLMLoss(target, pred)

# save test statistic               
t_file=open(output_folder+'t.txt', 'w')
t_file.write("%f\n"%(-2*nplm_loss_final))
t_file.close()

# save exec time                     
t_file=open(output_folder+'time.txt', 'w')
t_file.write("%f\n"%(t2-t1))
t_file.close()

# save monitoring metrics    
np.save(output_folder+'loss_history', loss_history)
np.save(output_folder+'nplm_loss_history', nplm_loss_history)
np.save(output_folder+'centroids_history', centroids_history)
np.save(output_folder+'widths_history', widths_history)
np.save(output_folder+'coeffs_history', coeffs_history)
print('Done')
