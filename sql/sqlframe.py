import os
import io
import sqlite3
import pandas as pd
import numpy as np
from .modifications import (SQLFrameModification, SQLFrameUpdate, SQLFrameAppend, 
                            SQLFrameModificationCatalog, 
                            SQLFrameModificationQueue)
from utils.sql.statements import compose_statement_insert_rows, compose_statement_create_table, compose_statement_select_rows_by_id
from ..datascience import get_type_string

# Adapter and converter so that we can save numpy arrays to SQLite
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter('ndarray', convert_array)

def get_sqlite_connection(path):
    try:
        connection = sqlite3.connect(path,detect_types=sqlite3.PARSE_DECLTYPES)
        return connection
    except sqlite3.Error as e:
        print(e)


class SQLFrameLoc():
    '''
    This class is a helper class so that SQL dataframes can do row indexing with .loc[...]
    '''
    def __init__(self,sqlframe):
        self.sqlframe = sqlframe

    def _format_selection(self,idx_and_col_selected):
        # Determine whether only row info, or also col info is given. If no col info is given, it's as if all
        # columns were selected with ":", so we make it a slice(None,None,None)
        is_there_col_info = isinstance(idx_and_col_selected,tuple)
        idx_selected = idx_and_col_selected[0] if is_there_col_info else idx_and_col_selected
        col_selected = idx_and_col_selected[1] if is_there_col_info else slice(None,None,None)
        # If selecting a slice, forward the slice. If no column selected, then all columns are selected
        if isinstance(idx_selected,slice): 
            idx_selected = self.sqlframe.index[idx_selected]
        if isinstance(col_selected,slice):
            col_selected = self.sqlframe.columns[col_selected]
        # Put selected rows (and cols, if any) into lists so that the function
        # compose_statement_select_rows_by_id can deal with them homogenously
        if isinstance(idx_selected,str): 
            idx_selected = [idx_selected]
        elif isinstance(idx_selected,pd.Index):
            idx_selected = list(idx_selected)
        if isinstance(col_selected,str):
            col_selected = [col_selected]
        elif isinstance(col_selected,pd.Index):
            col_selected = list(col_selected)
        return idx_selected, col_selected

    def _sql_loc(self,idx_selected,col_selected):
        '''
        This method grabs the values queried from the SQL table, and returns them as a pandas dataframe.
        '''
        # Execute selection
        connection = self.sqlframe.get_connection()
        cursor = connection.cursor()
        statement = compose_statement_select_rows_by_id(idx_selected,col_selected,self.sqlframe._index_name)
        cursor.execute(statement)
        selection = cursor.fetchall()
        connection.close()
        # Convert list of tuples to list of lists, and add the row indices
        if idx_selected is None:
            idx_selected = self.sqlframe.index 
        ## Return a dataframe rather than a list
        if col_selected is None:
            col_selected = list(self.sqlframe.columns)
        selection = [ [idx_row] +list(selection_row) for idx_row,selection_row in zip(idx_selected,selection)]
        df = pd.DataFrame(selection, columns = [self.sqlframe._index_name]+col_selected ).set_index(self.sqlframe._index_name)
        # Indexing the dataframe created with the original selection allows our return type (single value, pd.Series
        # or pd.DataFrame) to be consistent with the return type of pandas.DataFrame
        try:
            return df.loc[idx_selected,col_selected]
        except KeyError:
            return df

    def __getitem__(self,idx_and_col_selected):
        '''
        - key_list: list with keys. For example, could be ['Miguel','age'] to 
                    show the cell in row "Miguel" and column "age"
        - rows: single row name, or list with row names
        - columns:  single column name, or list with column names
        '''
        idx_selected, col_selected = self._format_selection(idx_and_col_selected)
        #return self.sqlframe._tmp_df.loc[idx_and_col_selected].combine_first(self._sql_loc(idx_and_col_selected))
        #return self._sql_loc(idx_selected,col_selected).combine_first(self.sqlframe._tmp_df.loc[idx_selected,col_selected]).loc[idx_and_col_selected]
        return self.sqlframe._tmp_df.loc[idx_selected,col_selected].combine_first(self._sql_loc(idx_selected,col_selected))#.loc[idx_and_col_selected]
        # TODO What if the SQL stores a "None"? In that case, combine_first doesn't replace the values in tmp_df by the values in the SQL table, and we get
        # NaN (what is in the table) instead of "None". Possible solution: filling the temporary dataframe with None rather than NaN

    def __setitem__(self,key,value):
        '''
        In order to either:
        - Update values with SQLFrame.loc[idx_selected,col_selected] = ...   , or
        - Create a new column with values
        '''
        idx_and_col_selected = key
        # Decide which type of modification we're dealing with, and append it to the modification queue
        idx_selected, col_selected = self._format_selection(idx_and_col_selected)
        missing_column = np.array([(col not in self.sqlframe._sql_columns) for col in col_selected]).any() # Is some column in the selection not in the SQL table?
        if not missing_column: # If not missing column, update
            # Make changes on the temporary dataframe
            __import__('pdb').set_trace()
            self.sqlframe._tmp_df.loc[idx_and_col_selected] = value
            # Record the changes in the modification queue
            self.sqlframe._modification_queue.add_record.update_values(idx_selected,col_selected)
        elif missing_column and len(col_selected) == 1:  # Create new columns one at a time only
            # Make changes on the temporary dataframe
            self.sqlframe._tmp_df.loc[idx_and_col_selected] = value
            # TODO Do something to handle missing values in new columns better. Can only create a new column if values for all rows are given
            self.sqlframe._modification_queue.add_record.add_single_column(col_selected)
        else:
            NotImplementedError('Allowed actions are to update values in existing columns, or to create a new single column.')



class SQLFrameIloc():
    def __init__(self,sqlframe):
        '''
        - sqlframe: this is the SQLFrame that owns this Iloc class
        '''
        self.sqlframe = sqlframe

    def __getitem__(self,idx_and_col_selected):
        # Check if only row info is given, or if also col info is given. If col info is given,
        # then it is as if all columns were selected with :, so slice(None,None,None)
        is_there_col_info = isinstance(idx_and_col_selected,tuple)
        idx_selected = idx_and_col_selected[0] if is_there_col_info else idx_and_col_selected
        col_selected = idx_and_col_selected[1] if is_there_col_info else slice(None,None,None)
        # Convert number indexing to string indexing
        idx_selected = self.sqlframe.index[idx_selected]
        col_selected = self.sqlframe.columns[col_selected]
        # Retrieve selected df with string indexing, using SQLFrame.loc[...] method
        return self.sqlframe.loc[idx_selected,col_selected]
        

class SQLFrame():
    '''
    This class implements a SQLite table (i.e. a database with a single table) with simple operations to insert new data, look up data, and delete
    data, either in the form of columns and rows. Columns can be accessed with square brackets, just like pandas dataframes.
    '''

    # NOTE A SQLFrame is a sqlite3 database with a single table. The name of the table can always be the same, and it can be "table"



    def __init__(self,path,base_dataframe,_create_from_scratch=True,_modification_queue=None,_tmp_df=None):
        '''
        If path is given, then init loads the database in the path. Otherwise, it creates a new database.
        - path: path for the database to be created
        - index: name of the column to use as index in the resulting database
        - columns: list with columns of the database to be created. The first column will be the index
        - types: dict with the types of the columns that will be used (necessary for adapters and converters, which will be useful for arrays).
                 The keys are the column names, and the values are the types
        - _create_from_scratch: private argument, do not use! (this simply allows a cleaner interface, so that creating sqlframes is done through
                    SQLFrame, and connecting to existing sqlframes is done through function connect_sqlframe)
        - _modification_queue: we allow the possibility to receive this, so that the return of operations can be a new SQLFrame
                               rather than making every modification in place.
        '''
        # TODO Improve how the arguments are given. Rather than _create_from_scratch , we could have an argument that can
        # take the values 'create' (new database), 'load' (from file), or something like that
        self.path = path
        base_df_index_name, base_df_columns = self.get_idx_and_col_names_info(base_dataframe)  # TODO Maybe if a database is given as a base, the values should be copied. Also, maybe the base dataframe should be the first argument, so that we can do  `sf = SQLFrame(df, path=some_path)`
        self._index_name = base_df_index_name
        # Attributes loc and iloc are used to implement .loc[...] and .iloc[...] indexing
        self.iloc = SQLFrameIloc(self)
        self.loc = SQLFrameLoc(self)
        # Modification queue will hold the changes (insertions, deletions and updates) that are intended on the SQLite database, until they are pushed
        self._modification_queue = SQLFrameModificationQueue(self) if _modification_queue is None else _modification_queue
        # Temporary dataframe will hold the values of the changes  until they are pushed to the SQLite database
        if _create_from_scratch:
            self._tmp_df = pd.DataFrame(columns=base_df_columns).set_index(self._index_name).replace({np.nan:None})  if _tmp_df is None else _tmp_df 
        else:
            self._tmp_df = self.create_mirror_dataframe()        # TODO If we are not creating the database from scratch , then we should create a temporary dataframe that matches the database dimensions

        # If we are to create a new database: 1) Raise error if it already exists 2) Create the main table (the latter is not necessary if 
        # database already exists)
        if _create_from_scratch:
            # Check if already exists
            if os.path.isfile(path):
                raise RuntimeError('File already exists in that location.')
            # Get the type of each column from the base dataframe
            base_df_types = self.get_col_types_info(base_dataframe)
            # Create the main table, with the same column names and types as the base dataframe we're replicating
            connection = self.get_connection()
            cursor = connection.cursor()
            statement = compose_statement_create_table(base_df_columns,base_df_types)
            cursor.execute(statement)
            connection.commit()
            connection.close()

    def get_idx_and_col_names_info(self, base_dataframe):
        '''
        Given a base dataframe whose structure we're trying to replicate, it returns
        the index name and the column names.
        '''
        index_name = base_dataframe.index.name
        columns = base_dataframe.reset_index().columns
        return index_name, columns

    def get_col_types_info(self, base_dataframe):
        '''
        Given a base dataframe whose structure we're trying to replicate, it returns
        the column types.
        '''
        columns = base_dataframe.reset_index().columns
        types = {}
        for col in columns:
            first_item = base_dataframe.reset_index().iloc[0].loc[col]
            col_type = get_type_string(first_item) # Assuming the whole column has the same type. This is a requirement of SQLite
            types[col] = col_type
        return types

    def create_mirror_dataframe(self):
        '''
        This method returns an empty dataframe that mirrors the database, in the sense that it has the same
        columns and index as it has.
        '''
        mirror_df = pd.DataFrame(columns=self._sql_columns).set_index(self._index_name) 
        mirror_df[self._index_name] = self._sql_index
        mirror_df = mirror_df.replace({np.nan:None}) 
        return mirror_df.set_index(self._index_name)

    # NOTE Not a property method because a property method self.connection suggests that a connection is an attribute that
    # is common to the entire class. However, this method creates a new connection, and self.get_connection() conveys that better
    def get_connection(self):
        '''
        Returns a connection to the database, except SQLite error. Implemented as a property method rather than as an attribute in __init__
        because for concurrency each thread must open its own connection  
        https://stackoverflow.com/questions/49918421/sqlite-concurrent-read-sqlite3-get-table
        '''
        return get_sqlite_connection(self.path)
    def get_cursor(self):
        # TODO Check if this method is really necessary, since we need the connection itself for most actions because we need to
        # close the connection after each action
        '''
        Returns a cursor to the database. Useful because most operations are done through the cursor directly, and the connection
        is not used.
        '''
        return self.get_connection().cursor()

    @property
    def shape(self):
        '''
        Returns a tuple with the number of rows and the number of columns, as in numpy and pandas
        '''
        return (len(self.index), len(self.columns))


    # TODO Improve the formatting of the returned tables (right now it's a tuple within a list or something like that...)
    @property
    def _tables(self):
        connection = self.get_connection()
        cursor = connection.cursor()
        statement = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        cursor.execute(statement)
        selection = cursor.fetchall()
        connection.close()
        return selection

    # TODO Extend/improve so that we append pandas dataframes and dataseries rather/in addition to dictionaries
    # TODO Insertions (and maybe updates?) should be made to temporary databases, so that we can choose when to save the changes manually
    # with a method such as SQLFrame.save() or something
    # TODO Better to change inplace, rather than return a SQLFrame. This is because we end up with several versions of the same SQLFrame,
    # but all point to the same database, so depending on which one we save we'll obtain one or another. Not good
    def append(self,df,verify_integrity=True):
        '''
        Appends data at the end of the temporary dataframe. Called append because it is similar to 
        pandas.DataFrame.append, and inplace because in contrast to pandas append, this one is inplace.
        - df: dataframe to append (syntax is the same as appending to a pandas.DataFrame)
        - verify_integrity: if True, raises ValueError on creating index with duplicates
        '''
        if verify_integrity: # NOTE In the original pandas, verify_integrity is False by default
            if df.index.isin(self.index).any():
                raise ValueError('Tried to append datapoint whose index is already in the database.') # ValueError because verify_integrity in pandas raises
                                                                                                      # a ValueError

        # Instead of changing the temporary dataframe and the queue of modifications in the current instance, 
        # return a new instance with these changed.

        # NOTE If we return self, we don't have to to append_inplace, since right now it isn't really inplace,
        # but rather append to the temporary dataframe
        self._modification_queue.add_record.append(df.index)
        self._tmp_df = self._tmp_df.append(df)

    # TODO Return _sql_types that returns the type string assigned to each column in the SQL dataframe

    def __getitem__(self,columns):
        '''
        This implements selection of columns with SQLFrame[...], as in pandas.DataFrame
        '''
        return self.loc[:,columns]
    def __setitem__(self,key,value):
        self.loc[:,key] = value

    def save(self):
        '''
        This method saves all temporary modifications to the SQLite database (insertions and updates) by pushing the 
        temporary dataframe into the database, and clearing the temporary dataframe.
        '''
        # 1. Push the changes to the SQLite database, iterating over the list of changes self._tmp_modifications
        self._modification_queue.push_all()
        # 2. Clean the temporary dataframe
        self._tmp_df = self.create_mirror_dataframe()
        # TODO After changing the columns, we need to save the temporary dataframe. That's a rule for now


    def set_index(self,index_name):
        '''
        This method changes the index to a different one, but it doesn't delete the previous index, 
        as pandas.DataFrame.set_index does
        '''
        self._index_name = index_name

    @property
    def _sql_index(self):
        '''
        Returns the index in the SQL
        '''
        connection = self.get_connection()
        cursor = connection.cursor()
        statement = 'SELECT ' + self._index_name + ' FROM my_table;'
        cursor.execute(statement)
        index = cursor.fetchall()
        connection.close()
        index = [idx[0] for idx in index]
        return pd.Index(data=index)
    @property
    def _sql_columns(self):
        '''
        This private method returns all columns, including the index name. Useful to compose SQLite 
        statements, but not coherent with pandas.columns, which doesn't include the index name in the columns
        '''
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info( 'my_table' );")
        info = cursor.fetchall()
        connection = self.get_connection()
        all_columns = [row[1] for row in info]
        return all_columns
    @property
    def index(self):
        return self._tmp_df.index
    @property
    def columns(self):
        all_columns = self._sql_columns
        columns = all_columns
        columns.remove(self._index_name)
        return pd.Index(data=columns)

    # TODO Write decorator to force to save after certain operations, such as adding a new column or changing the index. We could:
    # - Define an attribute such as SQLFrame.allowed_to_modify = True, which is changed to False if we do a sensitive operation
    # - Define a decorator to decorate all functions that do certain modifications, which checks whether the attribute is set to False,
    #   and if so throws an error
    
    def __repr__(self):
        n_rows = self.shape[0]
        pd_max_rows = pd.options.display.max_rows # Maximum number of rows printed by pandas to screen
        if pd_max_rows == 0:
            raise NotImplementedError('Printing to screen is not implemented when the maximum number of rows printed by pandas is 0.')
        # If our dataframe has more rows than pd_max_rows, we'll send to screen just a little bit more so that it gets printed with ... ,
        # and we'll manually set the number of columns and rows in the last line of output
        need_to_trim = True if n_rows > pd_max_rows else False
        if need_to_trim:
            n_head_and_tail = int(np.ceil((pd_max_rows+2)/2)) # Number of rows from the head and tail of the dataframe to print to screen
            idx_to_print = self.index[:n_head_and_tail].append(self.index[-n_head_and_tail:])  
            raw_string = self.loc[idx_to_print].__repr__()
            size_info = f'\n\n[{self.shape[0]} rows x {self.shape[1]} columns]'
            string = '\n'.join(raw_string.split('\n')[:-1]) + size_info
        else:
            string = self.loc[self.index].__repr__()
        return string

    def __str__(self):
        return self.__repr__()
        

# TODO Instead of passing index (i.e. the column name to use as index), create a different table in the SQLdatabase that stores only the 
# index name. Instead of having "my_table", this one could be called "data_table", and the index table could be called "index_table", 
# with a single columns of type string called "index_name"
def read_sqlframe(path,index):
    # Get column and type info
    connection = get_sqlite_connection(path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info( 'my_table' );")
    info = cursor.fetchall()
    columns = [row[1] for row in info]
    types = {row[1]:row[2] for row in info}
    # Create sqlframe
    df = pd.DataFrame(columns=columns).set_index(index)
    sf = SQLFrame(path=path,base_dataframe=df,_create_from_scratch=False)
    return sf
    




