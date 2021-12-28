import joblib
import pickle
import h5py
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sascorer
import sys
sys.path.append('../')
data = pd.read_hdf('/data/tp/VecVAE/zinc-1.h5', 'table')
smiles = data['smiles']
logp = data['logp']
qed = data['qed']
sas = data['sas']
for temp in ['CVAE', 'w2v', 'glove']:
    if temp == 'CVAE':
        from molecules.predicted_vae_model import VAE_prop
        #h5f = h5py.File('/data/tp/VecVAE/data/CVAE/per_all_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
        charset = open('data/CVAE/charset.pkl', 'rb')
        charset = pickle.load(charset)
    if temp == 'w2v':
        from molecules.predicted_vae_model_w2v import VAE_prop
        #h5f = h5py.File('/data/tp/VecVAE/data/w2v/per_all_w2v_35_w2_n1_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/w2v/predictor_vae_model_w2v_35_w2_n1_250000_0(5qed-sas)(std=1).h5'
        w2v_vector = open('data/w2v/w2v_vector_35_w2_n1.pkl', 'rb')
        word_vector = pickle.load(w2v_vector)
    if temp == 'glove':
        from molecules.predicted_vae_model_glove import VAE_prop
        #h5f = h5py.File('/data/tp/VecVAE/data/glove/per_all_glove_35_new_w2_250000.h5', 'r')
        modelname = '/data/tp/VecVAE/model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'
        glove_vector = open('data/glove/glove_vector_35_new_w2.pkl', 'rb')
        word_vector = pickle.load(glove_vector)

    model = VAE_prop()

    if os.path.isfile(modelname):
        model.load(35, 120, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    from optparse import OptionParser
    import rdkit
    from rdkit.Chem import Descriptors
    from rdkit.Chem import MolFromSmiles, MolToSmiles
    from rdkit.Chem import rdmolops
    import numpy as np
    import networkx as nx
    from molecules.util import vector120, get_w2v_vector

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    sas_values = []
    qed_values = []
    cycle_scores = []
    data = []

    for i in range(len(smiles)):
        if temp == 'CVAE':
            data.append(vector120(smiles[i], charset))
        else:
            data.append(get_w2v_vector(smiles[i], word_vector))
    
    latent_points = model.encoder.predict(np.array(data))

    for i in range(len(smiles)):
        qed_values.append(qed[i])
        sas_values.append(sas[i])

        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles[i]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)

    sas_normalized = (np.array(sas_values) - np.mean(sas_values)) / np.std(sas_values)
    qed_normalized = (np.array(qed_values) - np.mean(qed_values)) / np.std(qed_values)
    cycle_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
    # We store the results
    latent_file = 'data/' + temp + '/latent_features.txt'
    targets_file = 'data/' + temp + '/targets.txt'
    sas_file = 'data/' + temp + '/sas_values.txt'
    qed_file = 'data/' + temp + '/qed_values.txt'
    cycle_file = 'data/' + temp + '/cycle_values.txt'
    print(len(latent_points), len(sas_values), len(qed_values), len(cycle_scores))
    np.savetxt(latent_file, latent_points)
    targets = 5*qed_normalized - sas_normalized + cycle_normalized
    np.savetxt(targets_file, targets)
    np.savetxt(sas_file, np.array(sas_values))
    np.savetxt(qed_file, np.array(qed_values))
    np.savetxt(cycle_file, np.array(cycle_scores))

