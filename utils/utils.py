import numpy as np
import pandas as pd
from sklearn import preprocessing

class Data:
    def __init__(self):
        self.df = None
        self.data = None
        self.filename = None

    def __str__(self):
        try:
            return f'raw_data : {self.df.head()}'
        except:
            return 'import_df!!'

    def import_df(self, filename):
        self.filename = filename
        if self.filename.split('.')[-1] == 'csv':
            df = pd.read_csv(self.filename)
        self.df = df

        return self.df

    def df_to_data(self):
        self.data = self.df.values
        return self.data

class Preprocessing:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def cat_y_labeling(self, input_classes):
        le = preprocessing.LabelEncoder()
        le.fit(input_classes)
        y = le.transform(self.y)
        return y

    # def cat_X_labeling(self):
    #     oe = preprocessing.OrdinalEncoder()
    #     oe.fit