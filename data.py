# pylint: skip-file
""" file iterator for kaggle detect facial keypoints"""
import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
from pandas import read_csv
from sklearn.utils import shuffle
import pdb


FTRAIN = './data/training.csv'
FTEST = './data/test.csv'

class FileIter(DataIter):

    def _load(self, test=False, cols=None):
        """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
        Pass a list of *cols* if you're only interested in a subset of the
        target columns.
        """
        fname = FTEST if test else FTRAIN
        df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

        # The Image column has pixel values separated by space; convert
        # the values to numpy arrays:
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
        X = X.astype(np.float32)

        if not test:  # only FTRAIN has any target columns
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48  # scale target coordinates to [-1, 1]
            X, y = shuffle(X, y, random_state=42)  # shuffle train data
            y = y.astype(np.float32)
        else:
            y = None

        return X, y

    def _load2d(self, test=False, cols=None):
        X, y = self._load(test=test)
        X = X.reshape(-1, 1, 96, 96) 
        return X, y
    def __init__(self, eval_ratio = 0.2, is_val = False,
                 data_name = "data",
                 batch_size = 1,
                 label_name = "label"):
        self.eval_ratio = eval_ratio
        self.is_val = is_val
        self.data_name = data_name
        self.label_name = label_name
        self.size = 0
        self.batch_size = 1
        X, y = self._load2d() # X: (2140, 1, 96, 96), y: (2140, 30)
        if is_val:
            self.size = int(X.shape[0] * self.eval_ratio)
            self.X = X[-self.size : ]
            self.y = y[-self.size : ]
        else:
            self.size = int(X.shape[0] * (1 - self.eval_ratio))
            self.X = X[ : self.size]
            self.y = y[ : self.size]
        self.cursor = 0
        self.data, self.label = self._read()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_batch()
        return list(data.items()), list(label.items())

    def _read_batch(self):
        pdb.set_trace()
        if self.cursor + self.batch_size >= self.size:
            logger.info('data read out of bound')
            raise StopIteration
        
        begin = self.cursor
        self.cursor += self.batch_size
        return (self.X[begin : self.cursor], self.y[begin : self.cursor])

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def get_index(self):
        return self.cursor / self.batch_size

    def reset(self):
        self.cursor = 0

    def iter_next(self):
        if(self.cursor + self.batch_size < self.size):
            return True
        else:
            return False

    def getpad(self):
        return self.batch_size - self.size % self.batch_size

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            pdb.set_trace()
            self.data, self.label = self._read()
            return mx.io.DataBatch(data = self.data[0][1], label = self.label[0][1], 
                index = self.get_index(), pad = self.getpad())
            # return {self.data_name  :  self.data[0][1],
            #         self.label_name :  self.label[0][1]}
        else:
            raise StopIteration
