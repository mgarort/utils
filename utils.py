import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import json
import requests
from datetime import date
from functools import partial
import math


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
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol,canonical=True)
        return canon_smile
    except:
        print('Some conversions failed.')
        return None


def mol2morganfp(mol, nBits=1024, radius=3, return_bit_info=False):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bit_info)
    fp_array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    fp_array = fp_array.reshape(1,-1).astype(int)
    if return_bit_info:
        return fp_array, bit_info
    else:
        return fp_array


def robustify(function):
    """
    Function decorator that makes functions more robust and easier to integrate in pipelines, by adding the following rules:
    - If the input to the function is None, do not execute. Instead, return None.
    - If the function throws an error, do not stop execution. Instead, return None.
    - Only if the input is valid, and the function completes successfully, return the function's result.
    """
    def function_wrapper(x):
        if x is None or math.isnan(x):
            return x
        else:
            try:
                result = function(x)
                return result
            except Exception as e:
                print(e)
                return None
    return function_wrapper


class ConversionPipeline():
    """
    Used in my muticonverter expand_chem_df. It concatenates two functions together so that the intermediate 
    RDKit molecule object does not need to be stored in an iterable such as a Pandas Series. This way, we can bypass
    the memory leak detailed on the GitHub RDKit issue https://github.com/rdkit/rdkit/issues/3239
    """

    def __init__(self,input_function,output_function):
        self.input_function = input_function
        self.output_function = output_function

    def run(self,compound):
        mol = self.input_function(compound)
        if mol is not None:
            return self.output_function(mol)
        else:
            return None


def expand_chem_df(df,input_format,input_column,output_format,output_column,keep_None=True):
    """
    Given a dataframe where each row is a molecule and each column is a chemical property or chemical id,
    it returns the same dataframe with an additional column for a new chemical property or chemical id.
    
    df: Chemical dataframe to extend.
    input_format: string indicating molecular identifier used as input (smiles, canon smiles, inchi...).
    input_column: string indicating column name to use as input.
    output_format: string indicating molecular identifier or property used as output (smiles, canon smiles, inchi, inchi key, morgan fingerprints...).
    output_column: string indicating name of new column.
    keep_None: keep or delete null values None obtained either because the conversion failed or because the input was None to begin with. Default True.
    
    returns: extended dataframe.
    """

    # The input function will convert from a variety of molecular representations (SMILES, inchi...) to RDKit molecule object
    input_function_catalog = {'smiles': Chem.MolFromSmiles,
                              'inchi': Chem.MolFromInchi,
                             }
    # The output function will convert from RDKit molecule object to a molecular representation or property
    output_function_catalog = {'smiles': Chem.MolToSmiles,
                               'canonsmiles': partial(Chem.MolToSmiles, canonical=True),
                               'inchi': Chem.inchi.MolToInchi,
                               'inchikey': Chem.inchi.MolToInchiKey,
                               'morgan4fp': partial(mol2morganfp, radius=4),
                               'morgan3fp': partial(mol2morganfp, radius=3),
                               'morgan2fp': partial(mol2morganfp, radius=2),
                              }

    # Concatenate input and output function
    input_function = input_function_catalog[input_format]
    output_function = output_function_catalog[output_format]
    pipeline = ConversionPipeline(input_function,output_function)

    df_tmp = df.copy()
    df_tmp[output_column] = df_tmp[input_column].map(pipeline.run)
    if not keep_None:
        df_tmp = df_tmp.loc[~df_tmp[output_column].isnull()]
    return df_tmp


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

def smile2pdbfile(smile,filename,add_hydrogens=True,random_seed=2342,n_attempts=10):
    for idx_attempt in range(n_attempts):
        try:
            # Simple, meaningless expression to get a different random seed per attempt, 
            # but still keep the first random seed to that which was set manually
            random_seed = random_seed + idx_attempt * random_seed  
            mol = Chem.MolFromSmiles(smile)
            if add_hydrogens:
                mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol,randomSeed=random_seed)
            Chem.MolToPDBFile(mol,filename)
            success_creating_pdb = True
            return success_creating_pdb  # If the try statement gets to the end without an error, exit function and return True to indicate success
        except:
            pass
    success_creating_pdb = False
    return success_creating_pdb  # If none of the n_attempts succeeds, return False to indicate failure

def get_score_from_vina_logfile(logfile):
    with open(logfile, 'r') as f:
        counter_to_score = None
        for each_line in f:
            # Try to find the table header. Once found, count three lines to get the score
            if counter_to_score is not None:
                counter_to_score += 1
            if counter_to_score == 3:
                line_with_score = each_line
                break
            if 'mode |   affinity | dist from best mode' in each_line:
                counter_to_score = 0
        score = line_with_score.split()[1]
        return score

def get_pubchem_date(cid):
    """Get the creation date of a compound in PubChem from the compound id (cid)"""
    cid = str(int(cid))
    rest_request = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + cid + '/dates/json'
    resp = requests.get(rest_request).json()
    if 'Fault' in resp.keys():
        print(resp['Fault']['Message'])
        return 'None'
    else:
        creation_date = resp['InformationList']['Information'][0]['CreationDate']
        return date(year=creation_date['Year'],month=creation_date['Month'],day=creation_date['Day'])

def strlist2floatlist(strlist):
    """Function that converts a string representing a list of numbers to a list of floats.
     This is useful because sometimes pandas will save a column with strings of floats as strings."""
    strlist = strlist.strip('[').strip(']').split(',')
    floatlist = [float(elem) for elem in strlist]
    return floatlist
