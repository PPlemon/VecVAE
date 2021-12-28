import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import pickle
from molecules.predicted_vae_model_glove import VAE_prop
from rdkit import Chem
from rdkit.Chem import Draw
from molecules.util import vector120, get_w2v_vector
from scipy.spatial.distance import pdist
from keras.models import Model, load_model

def main():
    Ibuprofen = 'CC(C)Cc1ccc([C@H](C)C(=O)O)cc1'
    # Ibuprofen = 'O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1'
    w2v_vector = open('../data/glove/glove_vector_35_new_w2.pkl', 'rb')

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
            d0 = pdist(Y, metric='euclidean')[0]
            sims0.append(d0)
        sort0 = np.array(sims0).argsort()
        return [(id2word[i], sims0[i]) for i in sort0[:1]]

        # return v

    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []
    d = []
    modelname = '../model/glove/predictor_vae_model_glove_35_new_w2_250000_0(5qed-sas)(std=1).h5'

    if os.path.isfile(modelname):
        model.load(35, 120, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    Ibuprofen_encoded = get_w2v_vector(Ibuprofen, w2v_vector)
    Ibuprofen_encoded = np.array(Ibuprofen_encoded)

    Ibuprofen_latent = open('../data/glove/Ibuprofen_latent.pkl', 'rb')
    Ibuprofen_latent = pickle.load(Ibuprofen_latent)


    # Ibuprofen_latent = model.encoder.predict(Ibuprofen_encoded.reshape(1, width, 35))
    # Ibupro = open('../data/glove/Ibuprofen_latent.pkl', 'wb')
    # pickle.dump(Ibuprofen_latent, Ibupro)
    # print(Ibuprofen_latent)
    # Ibupro.close()


    for i in range(2000):
        Ibuprofen_x_latent = model.encoder.predict(Ibuprofen_encoded.reshape(1, width, 35))

        Ibuprofen_sampling_latent = Ibuprofen_x_latent[0]
        Y = np.vstack([Ibuprofen_latent, Ibuprofen_sampling_latent])
        d0 = pdist(Y)[0]

        sampled = model.decoder.predict(Ibuprofen_x_latent.reshape(1, latent_dim))[0]
        s = ''
        for j in range(len(sampled)):
            s += most_similar(sampled[j])[0][0]
        s = s.strip()

        if s == Ibuprofen:
            continue
        m = Chem.MolFromSmiles(s)
        print(s)
        if m != None:
            print(d0, s)
            if s not in result:
                result.append(s)
                f = '../data/picture/Ibuprofen_glove/3.0/' + str(d0) + '_Ibuprofen.png'
                Draw.MolToFile(m, f, size=(150, 100))
                d.append(d0)
            # print(d0, s)
    print(len(result))
    print(result)
    print(sum(d)/len(d))
if __name__ == '__main__':
    main()
