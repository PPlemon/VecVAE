import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
# 验证
fig = plt.figure(figsize=(9, 3))
spec = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1])
#spec.update(wspace=0., hspace=1.)
for f in range(3):
    if f == 0:
        from molecules.predicted_vae_model import VAE_prop
        h5f = h5py.File('/data/tp/VecVAE/data/CVAE/per_all_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
    if f == 1:
        from molecules.predicted_vae_model_w2v import VAE_prop
        h5f = h5py.File('/data/tp/VecVAE/data/w2v/per_all_w2v_35_w2_n1_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
    if f == 2:
        from molecules.predicted_vae_model_glove import VAE_prop
        h5f = h5py.File('/data/tp/VecVAE/data/glove/per_all_glove_35_new_w2_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
    data_train = h5f['smiles_train'][:]
    data_val = h5f['smiles_val'][:]
    data_test = h5f['smiles_test'][:]
    #logp_train = h5f['logp_train'][:]
    #logp_val = h5f['logp_val'][:]
    #logp_test = h5f['logp_test'][:]
    qed_train = h5f['qed_train'][:]
    qed_val = h5f['qed_val'][:]
    qed_test = h5f['qed_test'][:]
    sas_train = h5f['sas_train'][:]
    sas_val = h5f['sas_val'][:]
    sas_test = h5f['sas_test'][:]
    # print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
    # charset = h5f['charset'][:]

    length = len(data_test[0])
    charset = len(data_test[0][0])
    data = []
    logp = []
    qed = []
    sas = []
    for i in range(len(data_train)):
        data.append(data_train[i])
    #    logp.append(logp_train[i])
        qed.append(qed_train[i])
        sas.append(sas_train[i])
    for j in range(len(data_test)):
        data.append(data_test[j])
    #    logp.append(logp_test[j])
        qed.append(qed_test[j])
        sas.append(sas_test[j])
    for k in range(len(data_val)):
        data.append(data_val[k])
    #    logp.append(logp_val[k])
        qed.append(qed_val[k])
        sas.append(sas_val[k])
    data = np.array(data[:1000])
    #logp = np.array(logp)
    qed = np.array(qed[:1000])
    sas = np.array(sas[:1000])
    #print(len(data), len(logp), len(qed), len(sas))
    target = 5*qed-sas
    #print(qed[0], sas[0], target[0])
    h5f.close()
    model = VAE_prop()
    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)
    x_latent = model.encoder.predict(data)
    #pre_out = model.predictor.predict(x_latent)


    pca = PCA(n_components=50)
    x_latent_proj = pca.fit_transform(x_latent)

    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_))
    print(sum(pca.explained_variance_ratio_))
    del x_latent

    x_latent_proj = normalization(x_latent_proj)
    x = x_latent_proj[:, 0]
    y = x_latent_proj[:, 1]

    ax = fig.add_subplot(spec[0, f], projection='3d')
    #ax = fig.add_subplot(spec[0, f])
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_xlabel("X", fontsize=15)
    # ax.set_ylabel(" ", fontsize=15)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.get_yaxis().set_visible(False)
    ax.plot_trisurf(x, y, target, cmap='YlGnBu_r')
    #ax.plot(x, target)

# ax1 = fig.add_subplot(spec[0, 0], projection='3d')
# ax1.set_ylim(-0.05, 1.05)
# ax1.set_xlabel(" ", fontsize=15)
# ax1.set_ylabel("Y", fontsize=15)
# ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax1.get_xaxis().set_visible(False)
# sns.distplot(y, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
#
# ax2 = fig.add_subplot(spec[0, 1:])
# ax2.get_xaxis().set_visible(False)
# ax2.get_yaxis().set_visible(False)
# a = ax2.scatter(x, y, c=target, cmap='YlGnBu_r', marker='.', s=1)
#
# fig.colorbar(a, ax=ax2)
plt.savefig(fname="picture_1/latent_property.png",figsize=[9,3],bbox_inches='tight')
#plt.show()
