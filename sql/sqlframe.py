import os
import sqlite3
import pandas as pd

from utils.sql.statements import *

def get_sqlite_connection(path):
    try:
        connection = sqlite3.connect(path)
        return connection
    except sqlite3.Error as e:
        print(e)

class SQLFrameLoc():
    '''
    This class is a helper class so that SQL dataframes can do row indexing with .loc[...]
    '''
    def __init__(self,sqlframe):
        self.sqlframe = sqlframe
    def __getitem__(self,idx_and_col_selected):
        # TODO Allow indexing with lists of rows and columns, rather than single ones
        '''
        - key_list: list with keys. For example, could be ['Miguel','age'] to 
                    show the cell in row "Miguel" and column "age"
        - rows: single row name, or list with row names
        - columns:  single column name, or list with column names
        '''
        # Determine whether only row info, or also col info is given
        is_there_col_info = isinstance(idx_and_col_selected,tuple)
        idx_selected = idx_and_col_selected[0] if is_there_col_info else idx_and_col_selected
        col_selected = idx_and_col_selected[1] if is_there_col_info else None
        # Determine the return type according to the indices given to .loc[...].
        # Frame formatting across the row or col dimensions is maintained if a list
        # of rows or cols is given. If no col info is given, it is interpreted as
        # selecting all cols, so col frame formatting is maintained.
        # - loc[row, col] : return single value
        # - loc[row] : return pd.Series
        # - loc[[row]] : return pd.DataFrame
        # - loc[[row],[col]] : return pd.DataFrame
        should_keep_row_formatting = isinstance(idx_selected,list)
        should_keep_col_formatting = isinstance(col_selected,list) or (col_selected is None)
        # In Python, ":" indexing produces an object slice(None,None,None),
        # and it stands for selecting all
        if (isinstance(idx_selected,slice) and
            idx_selected.start is None and
            idx_selected.stop is None and
            idx_selected.step is None):
            idx_selected = None
        if (isinstance(col_selected,slice) and
            col_selected.start is None and
            col_selected.stop is None and
            col_selected.step is None):
            col_selected = None
        # Put selected rows (and cols, if any) into lists so that the function
        # compose_statement_select_rows_by_id can deal with them homogenously
        if (not isinstance(idx_selected,list)) and (idx_selected is not None):
            idx_selected = [idx_selected]
        if (not isinstance(col_selected,list)) and (col_selected is not None):
            col_selected = [col_selected]
        # Execute selection
        connection = self.sqlframe.get_connection()
        cursor = connection.cursor()
        statement = compose_statement_select_rows_by_id(idx_selected,col_selected,self.sqlframe._index_name)
        cursor.execute(statement)
        selection = cursor.fetchall()
        connection.close()
        # Convert list of tuples to list of lists, and add the row indices
        if idx_selected is None:
            idx_selected = self.sqlframe.index # TODO Implement SQLFrame.index method
        selection = [ [idx_row] +list(selection_row) for idx_row,selection_row in zip(idx_selected,selection)]
        ## Return a dataframe rather than a list
        #df = pd.DataFrame(selection, columns = [self.sqlframe._index_name]+col_selected ).set_index(self.sqlframe._index_name)
        ## TODO If returning a single column or a single row, check if it is better to return a pandas.Series
        #return df
        return selection
        

class SQLFrameIloc():
    def __init__(self,sqlframe):
        '''
        - sqlframe: this is the SQLFrame that owns this Iloc class
        '''
        self.sqlframe = sqlframe

    def __getitem__(self,rowid):
        pass
        

class SQLFrame():
    '''
    This class implements a SQLite table (i.e. a database with a single table) with simple operations to insert new data, look up data, and delete
    data, either in the form of columns and rows. Columns can be accessed with square brackets, just like pandas dataframes.
    '''

    # NOTE A SQLFrame is a sqlite3 database with a single table. The name of the table can always be the same, and it can be "table"
    # NOTE Commiting is to be done manually, not within the exit function

    def __init__(self,path,columns,index,types,_create_from_scratch=True):
        '''
        If path is given, then init loads the database in the path. Otherwise, it creates a new database.
        - path: path for the database to be created
        - index: name of the column to use as index in the resulting database
        - columns: list with columns of the database to be created. The first column will be the index
        - types: dict with the types of the columns that will be used (necessary for adapters and converters, which will be useful for arrays).
                 The keys are the column names, and the values are the types
        - _create_from_scratch: private argument, do not use! (this simply allows a cleaner interface, so that creating sqlframes is done through
                    SQLFrame, and connecting to existing sqlframes is done through function connect_sqlframe)
        '''
        self.path = path
        self._index_name = index 
        self.types = types # TODO Change so that type are determined automatically, rather than passed manually through a dictionary
        self.iloc = SQLFrameIloc(self)
        self.loc = SQLFrameLoc(self)
        # If we are to create a new database: 1) Raise error if it already exists 2) Create the main table (the latter is not necessary if 
        # database already exists)
        if _create_from_scratch:
            # Check if already exists
            if os.path.isfile(path):
                raise RuntimeError('File already exists in that location.')
            # Create the main table
            connection = self.get_connection()
            cursor = connection.cursor()
            statement = compose_statement_create_table(columns,self.types)
            cursor.execute(statement)
            connection.commit()
            connection.close()

    # NOTE Not made a property method because a property method self.connection suggests that a connection is an attribute 
    # that common to the entire class. However, this method creates a new connection, and self.get_connection() conveys that better
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
    def _n_rows(self):
        n_rows = len(self.index)
        return n_rows
    @property
    def _n_cols(self):
        n_cols = len(self.columns)
        return n_cols
    @property
    def shape(self):
        '''
        Returns a tuple with the number of rows and the number of columns, as in numpy and pandas
        '''
        return (self._n_rows, self._n_cols)


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

    # TODO Extend so that it works with more than a single row
    # TODO Extend/improve so that we append pandas dataframes and dataseries rather/in addition to dictionaries
    # TODO Insertions (and maybe updates?) should be made to temporary databases, so that we can choose when to save the changes manually
    # with a method such as SQLFrame.save() or something
    def append_inplace(self,row):
        '''
        Inserts rows at the end of the table. Called append because it is similar to pandas.DataFrame.append, and inplace because
        in contrast to pandas append, this one is inplace.
        - row: dictionary with column names as keys, and row values as values
        '''
        connection = self.get_connection()
        cursor = connection.cursor()
        statement = compose_statement_insert_rows(self._all_columns)
        values = tuple(row[column] for column in self._all_columns)
        cursor.execute(statement, values)
        connection.commit()

    # TODO Implement selection of columns, as in pandas dataframe
    def __getitem__(self,key):
        pass

    def set_index(self,index_name):
        '''
        This method changes the index to a different one, but it doesn't delete the previous index, 
        as pandas.DataFrame.set_index does
        '''
        self._index_name = index_name

    @property
    def index(self):
        connection = self.get_connection()
        cursor = connection.cursor()
        statement = 'SELECT ' + self._index_name + ' FROM my_table;'
        cursor.execute(statement)
        index = cursor.fetchall()
        connection.close()
        index = [idx[0] for idx in index]
        return index
    @property
    def _all_columns(self):
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
    def columns(self):
        all_columns = self._all_columns
        columns = all_columns
        columns.remove(self._index_name)
        return columns
        


def read_sqlframe(path,index):
    # Get column and type info
    connection = get_sqlite_connection(path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info( 'my_table' );")
    info = cursor.fetchall()
    columns = [row[1] for row in info]
    types = {row[1]:row[2] for row in info}
    # Create sqlframe
    sf = SQLFrame(path=path,columns=columns,index=index,types=types,_create_from_scratch=False)
    return sf
    




