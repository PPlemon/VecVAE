import rdkit
import pickle
import sascorer
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
# data = pd.read_hdf('../data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# print(len(smiles))

s = 'Cc1cc(=O)n(/C(Nc2ccc(Cl)c(Cl)c2)=[NH+]/C(C)C)s1'
mol = MolFromSmiles(s)
print(rdkit.Chem.QED.default(mol))
print(sascorer.calculateScore(mol))
# print(np.mean(np.array(logp)))
# print(np.mean(np.array(qed)))
# print(np.mean(np.array(sas)))
# logp_pre = []
# qed_pre = []
# sas_pre = []
#
# for i in range(len(smiles)):
#     s = smiles[i]
#     mol = MolFromSmiles(s)
#     logp_pre.append(Descriptors.MolLogP(mol))
#     qed_pre.append(rdkit.Chem.QED.default(mol))
#     sas_pre.append(sascorer.calculateScore(mol))
#
# print(np.mean(abs(np.array(logp) - np.array(logp_pre))))
# print(np.mean(abs(np.array(qed) - np.array(qed_pre))))
# print(np.mean(abs(np.array(sas) - np.array(sas_pre))))

