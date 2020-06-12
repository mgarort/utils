import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import json
import requests
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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
            return True  # If the try statement gets to the end without an error, exit function and return True to indicate success
        except:
            pass
    return False  # If none of the n_attempts succeeds, return False to indicate failure

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
    cid = str(cid)
    rest_request = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + cid + '/dates/json'
    resp = requests.get(rest_request).json()
    if 'Fault' in resp.keys():
        print(resp['Fault']['Message'])
        return 'None'
    else:
        creation_date = resp['InformationList']['Information'][0]['CreationDate']
        return date(year=creation_date['Year'],month=creation_date['Month'],day=creation_date['Day'])


def plot_precision_recall_curve(decision_function,X,Y,legend_location='upper right',filename=None,print_average_precision=False):
    """
    Simple function to plot a precision-recall curve.

    decision_function: from a sklearn classifier, for instance SVC().decision_function
    X: input variables for all datapoints
    Y: 0 and 1 labels for all datapoints
    legend_location: location of the legend in the plot. By default in the lower right.
    filename: if given, the plot will be saved to this location.
    returns: None
    """

    D = decision_function(X)
    prec, rec, _ = precision_recall_curve(Y, D)

    plt.figure()
    lw = 2
    if print_average_precision:
        ap = average_precision_score(Y, D)
        label_message = 'Precision recall curve (average precision = %0.2f)' % ap
    else:
        label_message = None
    plt.plot(rec, prec, color='darkorange',
             lw=lw, label=label_message)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision recall curve')
    plt.legend(loc=legend_location)
    if filename is not None:
        plt.savefig(filename)

# TODO Similar function for ROC, with area under the curve. Get it from the file predict_antibacterial_activity_roc.py, in the graph grammar playground
