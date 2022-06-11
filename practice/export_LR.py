import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# 파일 불러오기
filename= '../data/export.csv'
data = pd.read_csv(filename, encoding='utf-8')
data.drop([data.columns[0]],axis=1,inplace=True)
data.astype('int64')
print(data.info())
print()

y_column='한국 수출금액'
# column_cat1 = ['ocean_proximity']
# column_cat2 = ['...']

# null 값 제거
data.dropna(axis=0,inplace=True)
data.reset_index(drop=True,inplace=True)

# # 범주형 변수 처리(더미 개수 여러 개이면 아래 Task 반복)
# data_cat1 = data[column_cat1]
# data_cat1_dummies=pd.get_dummies(data_cat1, prefix=column_cat1[0])
# ## 더미 변수 결합 및 정제
# data = data.join(data_cat1_dummies)
# data.drop([column_cat1[0], data_cat1_dummies.columns[0]], axis=1, inplace=True)

# y, x : train,test split
y=data[y_column]
X=data.drop(y_column, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # standard scaling
# scaler=StandardScaler()
# X_train_std = scaler.fit_transform(X_train)
# X_test_std = scaler.transform(X_test)

#LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(f'LinearRegression Score: {score}')

#Lasso
model=Lasso(alpha=0.5)
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(f'Lasso Score: {score}')

#Ridge
model=Ridge(alpha=0.5)
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(f'Ridge Score: {score}')