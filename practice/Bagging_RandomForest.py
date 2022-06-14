import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# import data
df = pd.read_csv('../data/iris_short.csv')
print('raw data', df.head(5), sep='\n')
print()
data = df.values
 ## Set X,y
X = data[:,2:-1]
y = data[:,-1]
 ## preplocessing categorical data (y)
le=preprocessing.LabelEncoder()
classes_dict = {'versicolor':0, 'virginica':1}
le.fit(list(classes_dict.keys()))
y = le.transform(y)

# set variables
test_size = 0.3
dtc_max_depth = 2
n_estimators = 100
oob_score = True

if oob_score == False:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size) # train_test split
# Bagging with DecisionTreeClassifier
    model1 = BaggingClassifier(DecisionTreeClassifier(max_depth=dtc_max_depth), n_estimators=n_estimators,
                               oob_score=oob_score)
    model1.fit(X_train, y_train)
    # model.score(X_test, y_test)
    y_preds1 = model1.predict(X_test)
    print(f'Bagging_test_accuracy : {accuracy_score(y_test, y_preds1)}')
    print()
# RandomForestClassifier
    model2 = RandomForestClassifier(n_estimators=n_estimators)
    model2.fit(X_train, y_train)
    y_preds2 = model2.predict(X_test)
    print(f'RandomForestClassifier_test_accuracy : {accuracy_score(y_test, y_preds2)}')
    print()
elif oob_score == True:
# Bagging with DecisionTreeClassifier
    model1 = BaggingClassifier(DecisionTreeClassifier(max_depth=dtc_max_depth), n_estimators=n_estimators,
                               oob_score=oob_score)
    model1.fit(X, y)  # no train_test split
    oob_s1 = model1.oob_score_
    print(f'oob_score of Bagging : {oob_s1}')
# RandomForestClassifier
    model2 = RandomForestClassifier(n_estimators=n_estimators, oob_score=True)
    model2.fit(X, y)  # no train_test split
    oob_s2 = model2.oob_score_
    print(f'oob_score of RandomForestClassifier : {oob_s2}')