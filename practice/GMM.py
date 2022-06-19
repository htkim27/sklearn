import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

# randomly generate 4 clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.title("Raw Data")
plt.show()

# set variables
range_n_components = np.arange(1, 21)

# Select n_components(number of clusters) with AIC, BIC
models = [GaussianMixture(n, random_state=0).fit(X) for n in range_n_components]
bic_values = [m.bic(X) for m in models]
aic_values = [m.aic(X) for m in models]
# plt.plot(range_n_components, bic_values, label='BIC')
# plt.plot(range_n_components, aic_values, label='AIC')
# plt.xticks(range_n_components,range_n_components)
# plt.legend(loc='best')
# plt.xlabel('number of clusters')
# plt.show()

print(f'number of clusters when BIC is minimum : {np.argmin(bic_values)+1}')
print(f'number of clusters when AIC is minimum : {np.argmin(aic_values)+1}')
best_n = np.argmin(bic_values)+1 #or use np.argmin(aic_values)+1

# clustering with GaussianMixtureModels
gmm = GaussianMixture(n_components=best_n)
gmm.fit(X)
labels = gmm.predict(X)
plt.title('clustering with GMM')
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()

