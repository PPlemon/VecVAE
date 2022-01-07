import numpy as np
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import h5py
import os.path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
print(sys.path)
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops
import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
from scipy.spatial.distance import pdist
import sys
sys.path.append('../')
# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

# load model and data
temp = 'CVAE'

latent_file = '/data/tp/VecVAE/data/' + temp + '/latent_features.txt'
targets_file = temp + '/targets.txt'
qed_file = temp + '/qed_values.txt'
sas_file = temp + '/sas_values.txt'
cycle_file = temp + '/cycle_values.txt'
X = np.loadtxt(latent_file)
y = -np.loadtxt(targets_file)
y = y.reshape((-1, 1))

if temp == 'CVAE':
    from molecules.predicted_vae_model import VAE_prop
    #h5f = h5py.File('/data/tp/VecVAE/data/CVAE/per_all_250000.h5', 'r')
    modelname = '/data/tp/VecVAE/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
    charset = open('CVAE/charset.pkl', 'rb')
    charset = pickle.load(charset)
if temp == 'w2v':
    from molecules.predicted_vae_model_w2v import VAE_prop
    #h5f = h5py.File('/data/tp/VecVAE/data/w2v/per_all_w2v_35_w2_n1_250000.h5', 'r')
    modelname = '/data/tp/VecVAE/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
    word_vector_file = 'w2v/w2v_vector_35_w2_n1.pkl'
    distance = 'euclidean'
if temp == 'glove':
    from molecules.predicted_vae_model_glove import VAE_prop
    #h5f = h5py.File('/data/tp/VecVAE/data/glove/per_all_glove_35_new_w2_250000.h5', 'r')
    modelname = '/data/tp/VecVAE/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
    word_vector_file = 'glove/glove_vector_35_new_w2.pkl'
    distance = 'cityblock'
model = VAE_prop()
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)
#############

# molecule decode
if temp != 'CVAE':
    w2v_vector = open(word_vector_file, 'rb')
    w2v_vector = pickle.load(w2v_vector)
    word_vector = []
    id2word = []
    for key in w2v_vector:
        id2word.append(key)
        word_vector.append(w2v_vector[key])

def most_similar(w):
    sims0 = []
    for i in word_vector:
        Y = np.vstack([i, w])
        d0 = pdist(Y, metric=distance)[0]
        sims0.append(d0)
    sort0 = np.array(sims0).argsort()
    return [(id2word[i], sims0[i]) for i in sort0[:1]]

def decode_smiles_from_vector(vec):
    s = ''
    for j in range(len(vec)):
        s += most_similar(vec[j])[0][0]
    s = s.strip()
    return s
def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

###########
n = X.shape[0]
# m is the number of sample
m = 500
sort = y.argsort()[::-1]
#t = sort[:m]
#X = X[t]
#y = y[t]
permutation = np.random.choice(n, m, replace=False)

X_train = X[permutation, :][0: np.int(np.round(0.9 * m)), :]
X_test = X[permutation, :][np.int(np.round(0.9 * m)):, :]

y_train = y[permutation][0: np.int(np.round(0.9 * m))]
y_test = y[permutation][np.int(np.round(0.9 * m)):]
print(len(permutation), len(X_train))

qed = np.loadtxt(qed_file)
sas = np.loadtxt(sas_file)
cycle_scores = np.loadtxt(cycle_file)
# SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
# logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
# cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
all_valid_smiles = []
all_scores = []
iteration = 0
while iteration < 200:
    # We fit the GP
    np.random.seed(iteration * RANDOM_SEED)
    M = 32
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size=M,
                       max_iterations=20, learning_rate=0.0005)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test) ** 2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train) ** 2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale=np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(20, X[sort[iteration]], np.min(X_train, 0), np.max(X_train, 0))
    #next_inputs = sgp.batched_greedy_ei(20, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in range(20):
        all_vec = next_inputs[i]
        #print(all_vec)
        if temp == 'CVAE':
            sampled = model.decoder.predict(all_vec.reshape(1, 196)).argmax(axis=2)[0]
            s = decode_smiles_from_indexes(sampled, charset)
        else:
            sampled = model.decoder.predict(all_vec.reshape(1, 196))[0]
            s = decode_smiles_from_vector(sampled)
        m = Chem.MolFromSmiles(s)
        if m is not None:
            valid_smiles.append(s)
            new_features.append(all_vec)
        #else:
        #    valid_smiles.append(None)
        #    new_features.append(all_vec)

    print(len(valid_smiles), "molecules are found")
    #valid_smiles = valid_smiles[:50]
    #new_features = next_inputs[:50]
    #new_features = np.vstack(new_features)
    #save_object(valid_smiles, temp + "/result/valid_smiles{}.dat".format(iteration))
    all_valid_smiles.append(valid_smiles)
    scores = []
    for i in range(len(valid_smiles)):
        if valid_smiles[i] is not None:
            try:
                current_qed = rdkit.Chem.QED.default(MolFromSmiles(valid_smiles[i]))
                current_sas = sascorer.calculateScore(MolFromSmiles(valid_smiles[i]))
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[i]))))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([ len(j) for j in cycle_list ])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                current_cycle = -cycle_length

                print(current_qed, current_sas)

                sas_normalized = (current_sas - np.mean(sas)) / np.std(sas)
                qed_normalized = (current_qed - np.mean(qed)) / np.std(qed)
                cycle_normalized = (current_cycle - np.mean(cycle_scores)) / np.std(cycle_scores)
                score = 5*qed_normalized - sas_normalized + cycle_normalized
            except:
                score = -np.mean(y_train)
        else:
            score = -max(y)[0]
        scores.append(-score)  # target is always minused
    all_scores.append(scores)
    print(valid_smiles)
    print(scores)

    #save_object(scores, temp + "/result/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([X_train, new_features], 0)
        y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)

    iteration += 1

save_object(all_scores, temp + "/result/all_scores{}.dat".format(iteration))
save_object(all_valid_smiles, temp + "/result/all_valid_smiles{}.dat".format(iteration))
