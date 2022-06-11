import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# data
url= 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
df = pd.read_csv(url,
                     names=['buying','maint','doors','persons','lug_boot','safety','class'],
                     sep=",")

# encoding categorical variables
target_column='class'
features=df.columns.tolist()
features.remove(target_column)

## encoding features
o_encoder=OrdinalEncoder()
features_encoded = o_encoder.fit_transform(df[features])
encoded_df = pd.DataFrame(features_encoded,
                          columns=features)
## encoding target variable
l_encoder=LabelEncoder()
target_encoded = l_encoder.fit_transform(df[target_column])
encoded_df[target_column]=target_encoded

# CategrocialNB
## train_test split
data = encoded_df.values #data를 numpy로 추출
X=data[:,:-1]
y=data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# train_test
model = CategoricalNB()
model.fit(X_train, y_train)

y_predictions = model.predict(X_test)
print(f'test_accuracy_score : {accuracy_score(y_test,y_predictions)}\n')
print('Classification_Report',classification_report(y_test, y_predictions),sep='\n')