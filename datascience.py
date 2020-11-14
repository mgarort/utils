# This module will contain utilities related to the practice of data science in general, like cleaning up a dataset, converting between formats, splitting datasets, etc

import numpy as np
import pandas as pd
import json
from datetime import date
from functools import partial
import math
import h5py
import os
import sqlite3



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







def strlist2floatlist(strlist):
    """Function that converts a string representing a list of numbers to a list of floats.
     This is useful because sometimes pandas will save a column with strings of floats as strings."""
    strlist = strlist.strip('[').strip(']').split(',')
    floatlist = [float(elem) for elem in strlist]
    return floatlist

def is_hdf5_group(hdf5_elem):
    """ Given a HDF5 elem for a group or dataset, this function returns True if the elem refers to a group."""
    return isinstance(hdf5_elem, h5py.Group)


def is_hdf5_dataset(hdf5_elem):
    """ Given a HDF5 elem for a group or dataset, this function returns True if the elem refers to a dataset."""
    return isinstance(hdf5_elem, h5py.Dataset)


def hdf2dirs(hdf5_file, root_dir):
    """Function that creates a hierarchy of directories that copies a HDF5 file. HDF5 groups are made into directories, 
    and HDF5 datasets.
    - hdf5_file: path to the HDF5 file
    - root_dir: directory where we want to convert the HDF5 into diretories. Corresponds to the group '/' in the HDF5 file."""

    def group2dirs(group,group_dir):
        # Create directory for group 
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        # cd into that directory
        os.chdir(group_dir)
        # Convert to directorie and files
        for key in group.keys():
            elem = group[key]
            elem_path = os.path.join(group_dir, key)
            # For every group within the group: apply group2dirs to them
            if is_hdf5_group(elem):
                group2dirs(elem, elem_path)
            # For every dataset within the group: save the dataset as a numpy array/pickle/other format of your choice
            elif is_hdf5_dataset(elem):
                array = elem[...]
                array_path = elem_path + '.npy'
                np.save(array_path, array,allow_pickle=False)
        # cd ..
        os.chdir('..')

    with h5py.File(hdf5_file,'r') as hf: # TODO Maybe extract the iteration over the keys at the first level, and add a progress 
                                         #      bar over those to see progress while the file is processed
        group2dirs(hf,root_dir)


