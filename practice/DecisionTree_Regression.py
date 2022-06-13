import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

# import data
filename = '../data/Hitters_new.csv'
data = pd.read_csv(filename)


# shuffle data
data_np = data.values
np.random.shuffle(data_np)

# set features and target variable
X = data_np[:,1:3]
y = data_np[:,0]

# Regress with DecisionTree
model = DecisionTreeRegressor(max_depth=2)
model.fit(X,y)
print(f'score:{model.score(X,y)}')

# Visualizing
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3,3), dpi=300)
tree.plot_tree(model)
plt.title("Hitters' salary DecisionTreeRegressor")
plt.show()