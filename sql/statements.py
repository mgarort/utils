from ..datascience import get_type_string


def compose_statement_create_table(columns, types):
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
    statement += index_name + ' ' + types[index_name] #+ ' PRIMARY KEY' # 'index type PRIMARY_KEY, '
    # Add columns
    for each_column_name in column_names:
        statement += ', ' + each_column_name + ' ' + types[each_column_name] # ', column type'
    # Finish statement
    statement += ' );'
    return statement


##########################################
##### Statements to extend the table #####
##########################################

# TODO Extend it to compose a statement for multiple rows
# To be used by SQLFrameAppend
def compose_statement_insert_rows(columns):
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

# To be used by SQLFrameAddSingleColumn, as a pair
def compose_statement_add_single_column(column,column_type):
    return f'ALTER TABLE my_table ADD {column[0]} {column_type} ;'



##########################################
##### Statements to update the table #####
##########################################

def compose_statement_update(idx,col_selected,index_name):
    '''
    Compose statements to update the database one row at a time.
    '''
    statement = 'UPDATE my_table SET '
    col_selected = [f'{col} = ? ' for col in col_selected]
    statement += ', '.join(col_selected)
    statement += f"WHERE {index_name} IN ( '{idx}' );"
    return statement

    
########################################
##### Statements to view the table #####
########################################

# TODO To be used by sqlframe.[]  -->  __getitem__
def compose_statement_select_columns(col_selected):
    pass

# TODO To be used by sqlframe.iloc[]
def compose_statement_select_rows_by_number(n_rows):
    pass

def compose_statement_select_rows_by_id(idx_selected,col_selected,index_name):
    '''
    Compose statement to select rows by id (to be used by SQLFrame.loc[...])
    - idx_selected: list of strings, where each string is a row id
    - col_selected: list of strings, where each string is a col name

    If both rows and cols are selected, return statement of the form:
    'SELECT col_1, ... , col_m FROM my_table WHERE index_name IN ( row_1, ... , row_n );'

    Otherwise, return statement of the form
    'SELECT * FROM my_table WHERE index_name IN ( row_1, ... , row_n );'
    '''
    statement = 'SELECT '
    if col_selected is None:
        statement += '*'
    else:
        statement += ', '.join(col_selected)
    statement += ' FROM my_table'
    if idx_selected is None:
        statement += ' ;'
    else:
        statement += f' WHERE {index_name} IN ( '
        idx_selected = ["'" + row + "'" for row in idx_selected]
        statement += ', '.join(idx_selected)
        statement += ' );'
    return statement 

