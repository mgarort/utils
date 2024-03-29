from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from functools import partial
import numpy as np
import json
import requests
from datetime import date
import time


def get_max_min_coord(input_pdb):
    """Given an input pdb file with a ligand pose, this funtion examines the coordinates and returns a dictionary with the maximum and minimum
    x, y and z coordinates

    :input_pdb: filename of input pdb
    :returns: dictionary with keys max_x, min_x, max_y, min_y, max_z, min_z
    """
    # Get atomic coordinates from PDB file
    mol = Chem.MolFromPDBFile(input_pdb)
    conformer = mol.GetConformers()[0] # A PDB file probably gives rise to a single conformer, but in any case we're using only the first one
    pos = conformer.GetPositions()
    atomic_numbers = []
    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
    # Save the coordinates in the dictionary max_min_coord
    df_pos = pd.DataFrame(pos, columns=['x','y','z'])
    df_pos['atomic_number'] = atomic_numbers
    max_min_coord = {}
    max_min_coord['max_x'] = df_pos['x'].max()
    max_min_coord['min_x'] = df_pos['x'].min()
    max_min_coord['max_y'] = df_pos['y'].max()
    max_min_coord['min_y'] = df_pos['y'].min()
    max_min_coord['max_z'] = df_pos['z'].max()
    max_min_coord['min_z'] = df_pos['z'].min()
    
    return max_min_coord


def extract_best_pose(input_pdbqt, output_pdbqt):
    """ Given an input pdbqt file with several poses from an Autodock Vina run, copy the first one 
    in the file (which has the highest associated score) to a new, output pdbqt file """
    with open(input_pdbqt, 'r') as input_handle, open(output_pdbqt, 'w') as output_handle:
        for line in input_handle:
            output_handle.write(line)
            if line == 'ENDMDL\n':
                break


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


def smile2pdbfile(smile,filename,keep_hydrogens=True,random_seed=2342,n_attempts=10):
    for idx_attempt in range(n_attempts):
        try:
            # Simple, meaningless expression to get a different random seed per attempt,
            # but still keep the first random seed to that which was set manually
            random_seed = random_seed + idx_attempt * random_seed
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol) # Always add hydrogens in order to get a sensible 3D structure. Hydrogens can be removed later
            AllChem.EmbedMolecule(mol,randomSeed=random_seed)
            if not keep_hydrogens:
                mol = Chem.RemoveHs(mol)
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

def attempt_retrieving_pubchem_date(identifier, identifier_type='cid'):
    """
    Try to retrieve a date up to 5 times.
    """
    keep_trying = True
    num_tries = 0
    while keep_trying:
        if identifier_type == 'cid' or identifier_type == 'CID':
            cid = str(int(identifier))
            rest_request = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + cid + '/dates/json'
            resp = requests.get(rest_request)
        elif identifier_type  == 'smiles' or identifier_type == 'SMILES':
            smiles = identifier
            resp = requests.post('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/dates/JSON',
                                 headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                 data={'smiles':smiles})
        # If we have tried more than 5 times already, desist
        if num_tries > 5:
            keep_trying = False
        # If the server is busy or you have done too many requests (code 503) or if
        # the request has timed out (request 504), wait a little and try to get the date again
        elif resp.status_code == 503 or resp.status_code == 504:
            time.sleep(5)
            num_tries += 1
        # If the return is not any of the previous ones, either we have succeeded or the problem
        # cannot be fixed by trying again
        else:
            keep_trying = False
    return resp

def get_pubchem_date(identifier,identifier_type='cid'):
    """Get the creation date of a compound in PubChem from the compound id (cid)"""
    resp = attempt_retrieving_pubchem_date(identifier,identifier_type)
    # Because of an error in the connection the response may be incomplete or corrupted and the
    # JSON cannot be decoded. In this case, try again
    is_resp_decodable = False
    while not is_resp_decodable:
        try:
            resp = resp.json()
            is_resp_decodable = True
        except json.decoder.JSONDecodeError:
            resp = attempt_retrieving_pubchem_date(identifier,identifier_type)
    # If error return None. Otherwise return date
    if 'Fault' in resp.keys():
        print(resp['Fault']['Message'])
        return None
    else:
        creation_date = resp['InformationList']['Information'][0]['CreationDate']
        return date(year=creation_date['Year'],month=creation_date['Month'],day=creation_date['Day'])



def robustify(function): # This is not used anymore!! Maybe delete
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




# The following functions are probably obsolete. You should use ConversionPipeline to create a multiconverter that:
# 1. Is used in expand_chem_df
# 2. Can be used outside too, in order to replace the following functions
#
# Also, note that smile2pdbfile could be made part of the multiconverter, and it could just return True or False and write the pdb file on the side
#
#### FROM HERE

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


#### UP TO HERE
