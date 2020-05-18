import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs



def split_dataframe(df, split_fractions, shuffle=True, random_seed=None, filename_root=None):

    """This function takes as input a dataframe and splits it into training, validation and test sets, with
    the option to shuffle it, use a random seed or save them to file for the future.

    :df: dataframe to act on. Assumes that the label is the last column. If it is not a dataframe but an array, it gets converted
    :split_fractions: list or tuple with 3 fractions for training, validation and test. Must sum up to 1. 
    :shuffle: boolean. Default True.
    :random_seed: random seed for the shuffling. Default no random seed
    :filename_root: if you add a filename root, the splits obtained will be save with "filename_root"_train.tsv, etc.

    :returns: tuple with dataframes for training, validation and test.
    """

    assert sum(split_fractions) == 1, 'split fractions must sum up to 1'

    if type(df) == np.ndarray:
        df = pd.DataFrame(df)
    
    if shuffle:
        df = df.sample(frac=1,random_state=random_seed)

    n_samples = df.shape[0]
    n_train = int(n_samples*split_fractions[0])
    n_val = int(n_samples*split_fractions[1])
    n_test = int(n_samples*split_fractions[2])

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train+n_val]
    test = df.iloc[n_train+n_val:]

    if filename_root:
        train.to_csv(filename_root + '_train.tsv', sep='\t')
        val.to_csv(filename_root + '_val.tsv', sep='\t')
        test.to_csv(filename_root + '_test.tsv', sep='\t')

    return train, val, test

def smile2canonsmile(smile):
    mol = Chem.MolFromSmiles(smile)
    canon_smile = Chem.MolToSmiles(mol,canonical=True)
    return canon_smile

def smile2inchi(smile):
    mol = Chem.MolFromSmiles(smile)
    inchi = Chem.inchi.MolToInchi(mol)
    return inchi

def smile2inchikey(smile):
    mol = Chem.MolFromSmiles(smile)
    inchikey = Chem.inchi.MolToInchiKey(mol)
    return inchikey

def inchi2canonsmile(inchi):
    mol = Chem.inchi.MolFromInchi(inchi)
    smile = Chem.MolToSmiles(mol,canonical=True)
    return smile

def smile2fp(smile,nBits=1024):
    mol = Chem.MolFromSmiles(smile)
    bitInfo={}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, bitInfo=bitInfo)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, bitInfo

def smile2pdbfile(smile,filename,add_hydrogens=True):
    mol = Chem.MolFromSmiles(ris)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    Chem.MolToPDBFile(mol,filename)
