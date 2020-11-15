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
    def __getitem__(self,rows_and_cols):
        # TODO Allow indexing with lists of rows and columns, rather than single ones
        '''
        - key_list: list with keys. For example, could be ['Miguel','age'] to 
                    show the cell in row "Miguel" and column "age"
        - rows: single row name, or list with row names
        - columns:  single column name, or list with column names
        '''
        # Format the rows and cols selected. If single rows or cols are selected,
        # make them lists so that the compose statement function can deal with them
        rows_selected = rows_and_cols[0]
        if not isinstance(rows_selected,list): 
            rows_selected = [rows_selected]
        cols_selected = rows_and_cols[1]
        if not isinstance(cols_selected,list): 
            cols_selected = [cols_selected]
        # Execute selection
        connection = self.sqlframe.connection
        cursor = connection.cursor()
        statement = compose_statement_select_rows_by_id(rows_selected,cols_selected,self.sqlframe._index_name)
        cursor.execute(statement)
        selection = cursor.fetchall()
        connection.close()
        # Convert list of tuples to list of lists, and add the row indices
        selection = [ [idx] +list(row) for idx,row in zip(rows_selected,selection)]
        # Return a dataframe rather than a list
        df = pd.DataFrame(selection, columns = [self.sqlframe._index_name]+cols_selected ).set_index(self.sqlframe._index_name)
        # TODO If returning a single column or a single row, check if it is better to return a pandas.Series
        return df
        

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
        self.columns = columns # TODO self.columns should be a property method which returns the columns
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
            connection = self.connection
            cursor = connection.cursor()
            statement = compose_statement_create_table(self.columns,self.types)
            cursor.execute(statement)
            connection.commit()
            connection.close()

    # TODO Make a property method that returns the columns, which is self.columns

    @property
    def connection(self):
        '''
        Returns a connection to the database, except SQLite error. Implemented as a property method rather than as an attribute in __init__
        because for concurrency each thread must open its own connection  
        https://stackoverflow.com/questions/49918421/sqlite-concurrent-read-sqlite3-get-table
        '''
        return get_sqlite_connection(self.path)
    @property
    def cursor(self):
        # TODO Check if this method is really necessary, since we need the connection itself for most actions because we need to
        # close the connection after each action
        '''
        Returns a cursor to the database. Useful because most operations are done through the cursor directly, and the connection
        is not used.
        '''
        return self.connection.cursor()

    @property
    def _n_rows(self):
        cursor = self.cursor
        cursor.execute('SELECT count(*) from my_table;')
        n_rows = cursor.fetchone()[0]
        return n_rows
    # TODO Maybe change to simply counting self.columns? And self.columns could be a property method that returns the columns
    @property
    def _n_cols(self):
        cursor = self.cursor
        cursor.execute("SELECT count(*) FROM pragma_table_info( 'my_table' ) ;")
        n_cols = cursor.fetchone()[0]
        return n_cols
    @property
    def shape(self):
        '''
        Returns a tuple with the number of rows and the number of columns, as in numpy and pandas
        '''
        return (self._n_rows, self._n_cols)


    # TODO Improve the formatting of the returned tables (right now it's a tuple within a list or something like that...)
    @property
    def tables(self):
        cursor = self.cursor
        statement = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        cursor.execute(statement)
        return cursor.fetchall()

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
        connection = self.connection
        cursor = connection.cursor()
        statement = compose_statement_insert_rows(self.columns)
        values = tuple(row[column] for column in self.columns)
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
    




