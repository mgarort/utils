import h5py
import pandas as pd


def modification(foo):
    '''
    Decorator for all methods that modify the frame, which should
    check whether HFrame.modifiable is True before modifying anything.
    '''
    def wrapper(*args,**kwargs):
        self = args[0]
        if self.modifiable:
            foo(*args,*kwargs)
        else:
            raise RuntimeError('To modify HFrame, set HFrame.modifiable = True')
    return wrapper
            
    

        

class Iloc():

    def __init__(self,hframe):
        self.hframe = hframe

    def __getitem__(self,idx_number):
        '''
        For HFrame.iloc[idx_number] access.
        '''
        idx = self.hframe.index[idx_number]
        return self.hframe.loc[idx]

    @modification
    def __setitem__(self,idx_number,value):
        '''
        For HFrame.iloc[idx_number] = value setting of values.
        '''
        pass

    @property
    def modifiable(self):
        return self.hframe.modifiable


class Loc():

    def __init__(self,hframe):
        self.hframe = hframe

    def __getitem__(self,idx):
        '''
        For HFrame.loc[idx] access.
        '''
        return self.hframe._group_to_series(idx)
        
        

    @modification
    def __setitem__(self,idx,value):
        '''
        For HFrame.loc[idx] = value setting of values.
        '''
        pass

    @property
    def modifiable(self):
        return self.hframe.modifiable


class HFrame():
    '''
    We will imitate basic functionality of pandas dataframes with h5py so that we can save
    complicated and heavy stuff in cells.
    '''

    def __init__(self,path,columns,create_from_scratch=True):
        '''
        - path: path where the .hdf5 file will be saved
        - columns: the columns of the desired frame
        - create_from_scratch: whether we should create the HDF file, or load it
        '''
        # Attributes for the user
        self.columns = pd.Index(columns)
        # self.types = {key:value for key, value in zip(columns,types)}
        self.modifiable = False
        self.iloc = Iloc(self) # TODO
        self.loc = Loc(self) # TODO
        self.path = path
        # Attributes for internal usage
        if create_from_scratch:
            mode = 'w'
        else:
            mode = 'r+'
        self._hf = h5py.File(path,mode)
        self._index = pd.Index([])

    # To begin with, index will only be able to work with inchikeys. So we have index only for the ordering, but not
    # for using any kind of data as an index
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self,new_index):
        new_index = pd.Index(new_index)
        # Check that the length is the same
        if len(self.index) != len(new_index):
            raise ValueError('New index has different length from old index')
        # Check that every element in new_index is already in the old index
        any_new_index_not_in_old_index = (~new_index.isin(self.index)).any()
        if any_new_index_not_in_old_index:
            raise ValueError('One or more values in new index are not in the old index.')
        # Reorder the index by chaging the old index for the new index
        self._index = new_index

    # Methods for adding and removing data
    @modification
    def append(self,data):
        if isinstance(data, pd.DataFrame):
            for _, row in data:
                self._append_series(row)
        elif isinstance(data, pd.Series):
            self._append_series(data)
        else:
            raise ValueError('Data to append should be either pd.DataFrame or pd.Series')

    def _append_series(self,series):
        '''
        Each row in the frame is a HDF5 group. The name of the 
        group is the index of the row. The datasets in the group are the 
        column values in that row.
        '''
        index = series.name
        # Check that row with same index is not already in the HFrame
        if index in self.index:
            raise IndexError(f'Index {index} already in HFrame')
        # Create group for the series
        index = series.name  
        _ = self._hf.create_group(index)
        for column in series.index:
            _ = self._hf.create_dataset(index + '/' + column, data=series[column])
        # Add the series to the HFrame index
        self._index = self._index.append(pd.Index([index]))

    @modification
    def drop(self,axis=0):
        '''
        To drop rows (axis=0) or columns (axis=1). Note that we drop inplace
        '''
        pass


    # Methods to be used by HFrame.loc and HFrame.iloc
    def _group_to_series(self,index):
        '''
        Given a datapoint index, which is also the name of a group in the HDF5 file,
        convert that group to a dataseries.
        '''
        data = [self._hf[index][column][...] for column in self.columns]
        return pd.Series(data=data,index=self.columns,name=index)

    # Methods to close and clean up
    def __del__(self):
        '''
        When the method is garbage collected, make sure that it is closed.
        '''
        # Apparently we can test whether a file is open by h5py in this way
        # https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
        if self._hf.__bool__(): 
            self.close()

    def close(self):
        '''
        To manually close the handle to the HDF5 file once you're finished.
        '''
        self._hf.close()

    # Methods to be used by HFrame[]
    def __getitem__(self,column):
        '''
        For HFrame[column] access. 
        '''
        # You should iterate over the items in the HFrame and obtain the value of idx for each of the items
        data = [self._hf[index][column][...] for index in self.index]
        return pd.Series(data=data,index=self.index,name=column)


    @modification
    def __setitem__(self,column,value):
        '''
        For HFrame[column] = value  setting.
        '''
        pass


def read_hframe(path):

    hf = HFrame(path=path,columns=[],create_from_scratch=False)
    hf._index = pd.Index(hf._hf.keys())
    # Columns are obtained as the keys of the first group
    hf.columns = pd.Index(hf._hf[hf.index[0]].keys())

    return hf






