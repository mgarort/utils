# This module will contain utilities related to the practice of data science in general, like cleaning up a dataset, converting between formats, splitting datasets, etc

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


class SQLFrameLoc():
    '''
    This class is a helper class so that SQL dataframes can do row indexing with .loc[...]
    '''

    def __init__(self):


    def __getitem__(self,key):
        
def sql_connection(path):
   '''
   Connect to sqlite3 database and return connection, unless sqlite3 error.
   '''
    try:
        connection = sqlite3.connect(path)
        return connection
    except sqlite3.Error as e:
        print(e)


class SQLFrame():
    '''
    This class implements a SQLite table (i.e. a database with a single table) with simple operations to insert new data, look up data, and delete
    data, either in the form of columns and rows. Columns can be accessed with square brackets, just like pandas dataframes.
    '''

    # NOTE A SQLFrame is a sqlite3 database with a single table. The name of the table can always be the same, and it can be "table"
    # NOTE Commiting is to be done manually, not within the exit function

    def __init__(self,path,columns,types,_create_from_scratch=True):
        '''
        If path is given, then init loads the database in the path. Otherwise, it creates a new database.
        - path: path for the database to be created
        - index: index for the database to be created???? XXX Not sure if a good idea because maybe a database should be created one by one
        - columns: list with columns of the database to be created.
        - types: dict with the types of the columns that will be used (necessary for adapters and converters, which will be useful for arrays).
                 The keys are the column names, and the values are the types
        - _create_from_scratch: private argument, do not use! (this simply allows a cleaner interface, so that creating sqlframes is done through
                    SQLFrame, and connecting to existing sqlframes is done through function connect_sqlframe)
        '''
        self.path = path
        self.columns = columns
        self.types = types
        # Raise error if we're trying to create a new database from scratch but it already exists
        # (if the database already exists, we should connect to it rather than recreate it)
        if _create_from_scratch and os.path.isfile(path) :
            raise RuntimeError('File already exists in that location.')
        # Create the main table
        connection = sql_connection(self.path)
        cursor = connection.cursor()
        cursor.execute(self.create_table_statement)
        # TODO Create the table somewhere, probably in init
        #cursor.execute("CREATE TABLE table(id integer PRIMARY KEY, name text, salary real, department text, position text, hireDate text)")

    @property
    def connection(self):
        '''
        Returns a connection to the database. Implemented as a property method rather than as an attribute in __init__
        because for concurrency each thread must open its own connection  
        https://stackoverflow.com/questions/49918421/sqlite-concurrent-read-sqlite3-get-table
        '''
        return sql_connection(self.path)

    def create_table_statement(self):
        '''
        Composes statement to create table, of the form
        'CREATE TABLE table ( index type PRIMARY_KEY, column type, ... , column type, )'
        '''
        index_name = columns[0]
        columns_names = columns[1:]
        # Start statement
        statement = 'CREATE TABLE table ( '
        # Add index
        statement += index_name + ' ' + self.types + 'PRIMARY_KEY, ' # 'index type PRIMARY_KEY, '
        # Add columns
        for each_column_name in columns_names:
            statement += each_column_name + ' ' + types[each_column_name] + ', ' # 'column type, '
        # Finish statement
        statement += ' )'
        return statement

    # TODO To be used by append_inplace
    def insert_rows_statement(self):
        pass

    # TODO To be used by sqlframe.loc[]  -->  .loc.__getitem__
    def select_rows_statement(self):
        pass

    # TODO To be used by sqlframe[]  -->  __getitem__
    def select_columns_statement(self):
        pass

    # TODO
    def append_inplace(self,):
        '''
        Inserts rows at the end of the table. Called append because it is similar to pandas.DataFrame.append, and inplace because
        in contrast to pandas append, this one is inplace.
        '''
        connection = self.connection
        cursor = connection.cursor()
        pass

    def __getitem__(self,key):
        '''
        '''
        pass

    class loc

    # __enter__ and __exit__ are defined so that the database can be used within with statements,
    # and the connection is always closed upon exit
    def __enter__(self):
        return self
    def __exit__(self):
        self.connection.close()


def create_SQLFrame(path):
    '''
    This function creates a new SQLFrame on the location path, and returns it.
    '''
    # Check if file exists and complain if so
    if os.path.isfile(path):
        raise RuntimeError('File already exists in that location.')
    else:
        sqlframe = SQLFrame(path)




