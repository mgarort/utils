import abc
from utils.sql.statements import (compose_statement_insert_rows, compose_statement_create_table, 
                                  compose_statement_select_rows_by_id, compose_statement_update,
                                  compose_statement_add_single_column)
from ..datascience import get_type_string

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

    def __init__(self,sqlframe):
        '''
        - sqlframe: SQLFrame that the modifications refer to
        #- statement: string that is the SQL statement that needs to be executed to push the operation to the SQL database
        - info: necessary information to obtain the values to execute the statement (as in cursor.execute(statement,values))
                This attribute is not the values per se, but it is the info required to obtain the values from the 
                temporary dataframe. Depending on whether append, update, drop columns... the values will be obtained differently.
        '''
        self.sqlframe = sqlframe

    # TODO To be made an abstract method
    # XXX Not really needed if only SQLFrameModification is implemented, with statement and values
    @abc.abstractmethod
    def push(self,connection):
        '''
        Execute the SQLite modification held by the instance. Note that it receives a connection because in some modifications,
        we need to commit half the way (for instance, while adding a single column, in order to add the column before filling it)
        '''
        pass
        
class SQLFrameAppend(SQLFrameModification):
    '''
    For append, the attribute info contains the indices of the rows to be appended.
    '''
    def __init__(self,sqlframe,indices_appended):
        '''
        - indices_appended: indices of the rows that are appended
        '''
        super().__init__(sqlframe)
        print(indices_appended)
        self.indices_appended = indices_appended
        all_tmp_columns = self.sqlframe._tmp_df.reset_index().columns
        self.statement = compose_statement_insert_rows(all_tmp_columns) # TODO Could be list of statements if updating several rows. Change to statementS, maybe

    def push(self,connection):
        cursor = connection.cursor()
        # Append each row iteratively
        for idx in self.indices_appended:
            __import__('pdb').set_trace()
            values = self.sqlframe._tmp_df.loc[[idx]].reset_index().iloc[0].tolist()
            cursor.execute(self.statement,values)

class SQLFrameUpdate(SQLFrameModification):
    '''
    For update, the attribute info contains a tuble with the row indices and the column names
    that have been updated.
    '''
    def __init__(self,sqlframe,idx_selected,col_selected):
        '''
        - info: list or tuple with the row indices selected, and the col names selected for the update.
        '''
        super().__init__(sqlframe)
        self.idx_selected = idx_selected
        self.col_selected = col_selected
        self.statements = [compose_statement_update(idx,col_selected,self.sqlframe._index_name) for idx in idx_selected]
    def push(self,connection):
        cursor = connection.cursor()
        # Update values one row at a time
        idx_selected = self.idx_selected
        col_selected = self.col_selected
        for idx,statement in zip(self.idx_selected, self.statements):
            values = self.sqlframe._tmp_df.loc[idx, self.col_selected].tolist() 
            cursor.execute(statement,values)

class SQLFrameAddSingleColumn(SQLFrameModification):
    '''
    For when a single column is added with SQLFrame.loc[:,col] = ...   or with   SQLFrame[col] = ...
    '''
    def __init__(self,sqlframe,col_name,col_type):
        super().__init__(sqlframe)
        self.col_name = col_name
        self.add_column_statement = compose_statement_add_single_column(col_name,col_type)
        self.fill_column_statements = [compose_statement_update(idx,col_name,self.sqlframe._index_name) for idx in sqlframe.index]
    def push(self,connection):
        # Add the new column
        cursor = connection.cursor()
        __import__('pdb').set_trace()
        cursor.execute(self.add_column_statement)
        connection.commit()
        # Fill it
        for idx, statement in zip(self.sqlframe._tmp_df.index, self.fill_column_statements):
            value = [self.sqlframe._tmp_df.loc[idx,self.col_name[0]]]
            __import__('pdb').set_trace()
            cursor.execute(statement,value)

    


class SQLFrameModificationCatalog():
    '''
    This class is syntactic sugar. It will be used as an attribute called add_record in the modification queue, so that we can
    do SQLFrame._modification_queue.add_record.ACTION()
    '''
    
    def __init__(self,sqlframe):
        self.sqlframe = sqlframe

    def add_single_column(self,col_name):
        '''
        To record changes made with SQLFrame[...] (SQLFrame.__setitem__), where ... is a new column that doesn't exist 
        (if it existed we should update rather than insert it)
        If trying to insert several columns with this syntax, maybe raise a NotImplementedError
        '''
        col_type = get_type_string(self.sqlframe._tmp_df[col_name[0]].iloc[0])  # Get type of first element, assuming all the types are the same
        modification = SQLFrameAddSingleColumn(self.sqlframe,col_name,col_type)
        self.sqlframe._modification_queue.append(modification)
        # TODO After inserting columns you need to save if you want to keep working. That should be made a rule. Later we can think about how to avoid it.

    def append(self,indices):
        '''
        To record changes made with SQLFrame.append
        - indices
        '''
        modification = SQLFrameAppend(self.sqlframe,indices)
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

    def update_values(self,idx_selected,col_selected):
        '''
        To record changes made with SQLFrame.loc[...] (SQLFrame.loc.__setitem__) or with SQLFrame.iloc[...] (SQLFrame.iloc.__setitem__).
        ... will be recorded in "indices", but maybe change the name of "indices"
        '''
        modification = SQLFrameUpdate(self.sqlframe,idx_selected,col_selected) 
        self.sqlframe._modification_queue.append(modification)

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

    def __str__(self):
        print(self.queue)

    def push_all(self):
        '''
        This method saves all temporary modifications to the SQLite database
        and cleans the modifications queue.
        '''
        connection = self.sqlframe.get_connection()
        for modification in self.queue:
            modification.push(connection)
        connection.commit()
        connection.close()
        self.queue = []
