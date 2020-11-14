def create_table_statement(columns, types):
    '''
    Composes statement to create table, of the form
    'CREATE TABLE my_table ( index type PRIMARY KEY, column type, ... , column type, );'
    - columns: list of str with the names of the columns. The first column is 
               assumed to work as index
    - types: dictionary with the type of each column
    '''
    index_name = columns[0]
    column_names = columns[1:]
    # Start statement ("my_table" is an arbitrary name. Note that "table" cannot be used because it's a reserved word)
    statement = 'CREATE TABLE my_table ( '
    # Add index
    statement += index_name + ' ' + types[index_name] + ' PRIMARY KEY' # 'index type PRIMARY_KEY, '
    # Add columns
    for each_column_name in column_names:
        statement += ', ' + each_column_name + ' ' + types[each_column_name] # ', column type'
    # Finish statement
    statement += ' );'
    return statement

# TODO To be used by append_inplace
# TODO Extend it to compose a statement for multiple rows
def insert_rows_statement(columns):
    '''
    Composes statement to insert a single row, of the form
    'INSERT INTO my_table ( column_1, ... , column_2 ) VALUES ( ?, ..., ? );'
    - columns: list of str with the names of the columns
    '''
    # Start statement
    statement = 'INSERT INTO my_table ( '
    # Add column names (including index, which is the first column)
    statement += ', '.join(columns)
    # Continue statement
    statement += ' ) VALUES ( '
    # Add the interrogation signs
    n_cols = len(columns)
    statement += ', '.join(['?']*n_cols) # '?, ?, ... , ?'
    # Finish statement
    statement += ' );'
    return statement
    

# TODO To be used by sqlframe.loc[]  -->  .loc.__getitem__
def select_rows_statement(self):
    pass

# TODO To be used by sqlframe[]  -->  __getitem__
def select_columns_statement(self):
    pass
