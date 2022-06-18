import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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
            print('raw_data',df.head(),sep='\n')
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

class GridSearch:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None

    def gs_best_model(self, estimator, param_grid):
        # Instantiate GridSearchCV as grid_reg
        grid_clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, n_jobs=-1)

        # Fit grid_reg on X_train and y_train
        grid_clf.fit(self.X_train, self.y_train)
        best_params = grid_clf.best_params_
        print(f'best_params : {best_params}')

        # Extract best estimator
        best_model = grid_clf.best_estimator_

        # Extract best score
        best_score = grid_clf.best_score_
        self.best_model = best_model

        # Print best score
        print("grid_search score: {:.3f}".format(best_score))

        # Return best model
        return best_model

    def print_test_score(self):

        # Predict test set labels
        best_model = self.best_model
        y_pred = best_model.predict(self.X_test)
        # Compute accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Print accuracy
        print('Test score: {:.3f}'.format(accuracy))