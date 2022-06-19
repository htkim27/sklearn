import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import validation

# import data
filename = '../data/pca_data.csv'
df = pd.read_csv(filename,header=None)
data = df.values
print('raw data', df, sep='\n')
print()

# set variables
n_components=1

# Using sklearn
pca = PCA(n_components = n_components)
new_data = pca.fit_transform(data)
new_data = validation.column_or_1d(new_data)
print(f'result of PCA : {new_data}')

# Without sklearn
# step1. mean centering
means = data.mean(axis=0)
data = data - means
# print('centered data', data, sep='\n')
# print()

# step2. covariance matrix of data
cov_matrix = np.cov(data.T)

# step3. eigendecomposition : to get eigen vectors and eigen values from cov_matrix
eigvalues, eigvectors = np.linalg.eig(cov_matrix)
# print(f'(eigenvalue,eigenvector) : {list(zip(eigvalues, list(eigvectors)))}')
# print()

# step4. reset data values along the new axis
idx = np.argmax(eigvalues)
new_axis = eigvectors[:,idx]
# set new value of datapoint by projection
new_data = np.dot(data, new_axis)
print(f'new_data from PCA(2D -> 1D) : {new_data}')