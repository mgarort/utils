import abc
from utils.sql.statements import compose_statement_insert_rows, compose_statement_create_table, compose_statement_select_rows_by_id

# TODO Each of the following must implement:
# - Holding all the required information (mainly row/index and column information)
# - push_to_SQLite method, that saves the current modification to the SQL table
# This way, SQLiteFrame.save() can simply iterate over the list of temporary modifications, and push each of them


#class SQLFrameInsertSingleColumn(SQLFrameModification):
#    pass
#
#class SQLFrameDropRow(SQLFrameModification):
#    pass
#
#class SQLFrameDropColumn(SQLFrameModification):
#    pass
#
#class SLQFrameUpdateValues(SQLFrameModification):
#    pass


# TODO Maybe not this, but an attribute called SQLFrame.modifications to which you can .insert, .delete and .update
class SQLFrameModification(abc.ABC):
    '''
    This class holds a single modification to the SQLite database. The nature of the modification
    (insert, delete, update, etc) is described in self.statement and self.values. These are received
    from SQLFrameModificationCatalog.
    '''
    # NOTE The following is a primitive version of how appending rows could be implemented
    #connection = self.get_connection()
    #cursor = connection.cursor()
    #statement = compose_statement_insert_rows(self._all_columns)
    #values = tuple(row[column] for column in self._all_columns)
    #cursor.execute(statement, values)
    #connection.commit()

    def __init__(self,sqlframe,info):
        '''
        - sqlframe: SQLFrame that the modifications refer to
        #- statement: string that is the SQL statement that needs to be executed to push the operation to the SQL database
        - info: necessary information to obtain the values to execute the statement (as in cursor.execute(statement,values))
                This attribute is not the values per se, but it is the info required to obtain the values from the 
                temporary dataframe. Depending on whether append, update, drop columns... the values will be obtained differently.
        '''
        self.sqlframe = sqlframe
        self.info = info # TODO Maybe not necessary to save values because they will be in the dataframe
                             #      Maybe rather than the actual values it should be the rows and indices of the temporal dataframe

    # TODO To be made an abstract method
    # XXX Not really needed if only SQLFrameModification is implemented, with statement and values
    @abc.abstractmethod
    def push(self,cursor):
        '''
        Execute the SQLite modification held by the instance. Note that it receives only a cursor because the connection is opened,
        committed and closed in SQLFrameModificationQueue.push_all()
        '''
        pass
        
class SQLFrameAppend(SQLFrameModification):
    '''
    For append, the attribute info contains the indices of the rows to be appended.
    '''
    def __init__(self,sqlframe,info):
        '''
        - info: indices of the rows that are appended
        '''
        super().__init__(sqlframe,info)
        all_tmp_columns = self.sqlframe._tmp_df.reset_index().columns
        self.statement = compose_statement_insert_rows(all_tmp_columns) # TODO Could be list of statements if updating several rows. Change to statementS, maybe

    def push(self,cursor):
        # Append each row iteratively
        indices = self.info
        for idx in indices:
            values = self.sqlframe._tmp_df.loc[[idx]].reset_index().iloc[0].tolist()
            cursor.execute(self.statement,values)

class SQLFrameUpdate(SQLFrameModification):
    '''
    For update, the attribute info contains a tuble with the row indices and the column names
    that have been updated.
    '''
    def push(self,cursor):
        idx_selected, col_selected = info
        for idx,statement in zip(idx_selected,self.statement):
            values = self.sqlframe._tmp_df.loc[idx,col_selected] # TODO Check that the final type is correct
            cursor.execute(statement,values)
        
        # TODO Current statement should be in self.statement. Execute it

    


class SQLFrameModificationCatalog():
    # NOTE Not necessary to save values because they are in the frame with modifications 
    
    def __init__(self,sqlframe):
        self.sqlframe = sqlframe

    def insert_single_column(self,column_name):
        '''
        To record changes made with SQLFrame[...] (SQLFrame.__setitem__), where ... is a new column that doesn't exist 
        (if it existed we should update rather than insert it)
        If trying to insert several columns with this syntax, maybe raise a NotImplementedError
        '''
        pass

        # TODO After inserting columns you need to save if you want to keep working. That's a rule for now. Later we can think about how to avoid it.

    def append(self,indices):
        '''
        To record changes made with SQLFrame.append
        - indices
        '''

        #'''
        #Composes statement to insert a single row, of the form
        #'INSERT INTO my_table ( column_1, ... , column_2 ) VALUES ( ?, ..., ? );'
        #- columns: list of str with the names of the columns
        #'''
        ## Start statement
        #statement = 'INSERT INTO my_table ( '
        ## Add column names (including index, which is the first column)
        #statement += ', '.join(columns)
        ## Continue statement
        #statement += ' ) VALUES ( '
        ## Add the interrogation signs
        #n_cols = len(columns)
        #statement += ', '.join(['?']*n_cols) # '?, ?, ... , ?'
        ## Finish statement
        #statement += ' );'
        #return statement

        modification = SQLFrameAppend(self.sqlframe,indices)
        
        # TODO Compose statement and save it in modification
        # TODO Save values in modification too if needed. If not, save just None to reflect that the statement is complete and no values are needed
        self.sqlframe._modification_queue.append(modification)

    def drop_row(self,index_name):
        '''
        To record changes made with SQLFrame.drop(axis=0)
        '''
        pass

    def drop_column(self,column_name):
        '''
        To record changes made with SQLFrame.drop(axis=1)
        '''

    def update_values(self,indices):
        '''
        To record changes made with SQLFrame.loc[...] (SQLFrame.loc.__setitem__) or with SQLFrame.iloc[...] (SQLFrame.iloc.__setitem__).
        ... will be recorded in "indices", but maybe change the name of "indices"
        '''

class SQLFrameModificationQueue():
    '''
    The idea is that each __setitem__ will do self.modification_queue.add_record.append/drop_row/updte_values etc.
    Then at the end of a certain operation, SQLFrame.save can use self.modification_queue.push_all
    '''

    def __init__(self,sqlframe):
        self.sqlframe = sqlframe
        self.queue = []

    # XXX Maybe delete if not used
    def __getitem__(self,idx):
        return self.queue[idx]

    def append(self,item):
        self.queue.append(item)

    @property
    def add_record(self):
        return SQLFrameModificationCatalog(self.sqlframe)

    def push_all(self):
        connection = self.sqlframe.get_connection()
        cursor = connection.cursor()
        for modification in self.queue:
            modification.push(cursor)
        connection.commit()
        connection.close()
