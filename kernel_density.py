import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import os
import h5py
#from molecules.predicted_vae_model import VAE_prop
#from molecules.predicted_vae_model_w2v import VAE_prop
from molecules.predicted_vae_model_glove import VAE_prop
import matplotlib.pyplot as plt
import seaborn as sns

# 正态分布函数
#def normfun(x, mu, sigma):
#    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
#    return pdf

latent_dim = 196
# 画正态分布图
#h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
#h5f = h5py.File('/data/tp/data/per_all_w2v_35_w2_n1_250000.h5', 'r')
h5f = h5py.File('/data/tp/data/per_all_glove_35_new_w2_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
# data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
logp_test = h5f['logp_test'][:]
# print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
#charset2 = h5f['charset'][:]
#charset1 = []
#for i in charset2:
#    charset1.append(i.decode())
model = VAE_prop()
length = len(data_test[0])
charset = len(data_test[0][0])
#modelname = '/data/tp/data/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
#modelname = '/data/tp/data/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
modelname = '/data/tp/data/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'

if os.path.isfile(modelname):
    model.load(charset, length, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

x_latent = model.encoder.predict(data_train)
latent = [[]for _ in range(latent_dim)]

#def normalization(data):
#    _range = np.max(data) - np.min(data)
#    return (data - np.min(data)) / _range

#x_latent = normalization(x_latent)

m = []
mean = []
var = []
std = []
for i in range(latent_dim):
    for j in range(len(data_train)):
        latent[i].append(x_latent[j][i])
    m.append(max(latent[i]))
    mean.append(np.mean(latent[i]))
    var.append(np.var(latent[i]))
    std.append(np.std(latent[i]))
print(np.mean(mean), np.mean(var), np.mean(std))

#fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot()
#ax.set_xlabel("Z (unstandardized)", fontsize=15)
#ax.set_ylabel("Normalized Frequency", fontsize=15)

#ax.set_xlim(-1.0, 1.0)
#ax.set_ylim(0, 30)
#plt.xticks([-1.0, -0.5, 0, 0.5, 1.0], fontsize=15)
#plt.yticks([0, 5, 10, 15, 20, 25, 30], fontsize=15)

#for i in range(196):
#    x = latent[i]
#    sns.distplot(x, ax=ax, hist=False, kde_kws={"shade": False, "color": (random.random(),
#                                                                          random.random(),
#                                                                          random.random())})
#plt.savefig(fname="picture_1/predictor_vae_glove_kernel(5qed-sas)(std=1).png",figsize=[6,6],bbox_inches='tight')
