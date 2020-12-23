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
        pass

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

    def __init__(self,path,columns):
        '''
        - path: path where the .hdf5 file will be saved
        - columns: the columns of the desired frame
        - types: the types of the columns
        '''
        self.columns = columns
        # self.types = {key:value for key, value in zip(columns,types)}
        self.modifiable = False
        self.iloc = Iloc(self) # TODO
        self.loc = Loc(self) # TODO
        self.path = path
        self._hf = h5py.File(path,'w')


    def __del__(self):
        '''
        When the method is garbage collected, make sure that it is closed.
        '''
        # Apparently we can test whether a file is open by h5py in this way
        # https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
        if not self._hf.__bool__(): 
            self.close()

    def close(self):
        '''
        To manually close the handle to the HDF5 file once you're finished.
        '''
        self._hf.close()


    def __getitem__(self,column):
        '''
        For HFrame[column] access. 
        '''
        # You should iterate over the items in the HFrame and obtain the value of idx for each of the items
        pass


    @modification
    def __setitem__(self,column,value):
        '''
        For HFrame[column] = value  setting.
        '''
        pass

    @modification
    def append(self,data):
        if isinstance(data, pd.DataFrame):
            for _, row in data:
                self._append_series(row)
        elif isinstance(data, pd.Series):
            self._append_series(data)
        else:
            raise ValueError('Data to append should be either pd.DataFrame or pd.Series')

    @modification
    def drop(self,axis=0):
        '''
        To drop rows (axis=0) or columns (axis=1). Note that we drop inplace
        '''

    def _append_series(self,series):
        '''
        Each row in the frame is a HDF5 group. The name of the 
        group is the index of the row. The datasets in the group are the 
        column values in that row.
        '''
        index = series.name  
        _ = self._hf.create_group(index)
        for column in series.index:
            _ = self._hf.create_dataset(index + '/' + column, data=series[column])

    def _group_to_series(self,index):
        '''
        Given a datapoint index, which is also the name of a group in the HDF5 file,
        convert that group to a dataseries.
        '''
        series = pd.Series(index=self.columns,name=index)
        for column in series.index:
            series[column] = self._hf[index][column][...]
        return series




