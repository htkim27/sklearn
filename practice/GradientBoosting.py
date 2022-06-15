import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from utils import utils
from sklearn.utils import validation
from sklearn.metrics import mean_squared_error as MSE

# import data
df = pd.read_csv('../data/bike_rentals.csv')
print(f'raw data {df.head()}')

# set variables
test_size = 0.2
target_column = 'cnt'
grid_search_clf_params = {'max_depth':list(range(2,8)),
                          'n_estimators':[3,30,300],
                          'learning_rate':[0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.5,1.0],
                          'subsample':[1, 0.9, 0.8, 0.7, 0.6, 0.5]
                          }

# preprocess data to X,y
y=df.loc[:,[target_column]].values
X=df.drop(target_column, axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

# grid search를 위해 y값을 1d array로 변환
print(y_train.shape)
y_train_gs = validation.column_or_1d(y_train)
y_test_gs = validation.column_or_1d(y_test)
print(y_train_gs.shape)

# GridSearch
gs = utils.GridSearch(X_train, X_test, y_train_gs, y_test_gs)
best_model = gs.gs_best_model(GradientBoostingRegressor(), grid_search_clf_params)

# train_test with best_model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
RMSE = MSE(y_test, y_pred)**0.5
print(f'RMSE : {RMSE}')
