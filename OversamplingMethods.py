import imblearn
from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from numpy import where
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# visualizing
class OversamplingDriver:
    def __init__(self, X, y, method, sampling_strategy='auto'):
        self.X=X
        self.y=y
        self.method=method
        self.sampling_strategy=sampling_strategy

    def over_sampling(self):
        if self.method == 'SMOTE':
            os_m = SMOTE(sampling_strategy=self.sampling_strategy)
            X_o, y_o = os_m.fit_resample(X=self.X, y=self.y)
            return X_o, y_o

        elif self.method == 'Borderline_SMOTE':
            os_m = BorderlineSMOTE(sampling_strategy=self.sampling_strategy)
            X, y = os_m.fit_resample(X=self.X, y=self.y)
            return X, y

        elif self.method == 'ADASYN':
            os_m = ADASYN(sampling_strategy=self.sampling_strategy)
            X, y = os_m.fit_resample(X=self.X, y=self.y)
            return X, y

    def over_under_sampling(self, u_sampling_strategy=0.5):
        if self.method == 'SMOTE':
            os_m = SMOTE(sampling_strategy=self.sampling_strategy)
            us_m = RandomUnderSampler(sampling_strategy=u_sampling_strategy)
            X_o, y_o = os_m.fit_resample(X=self.X, y=self.y)
            X_ou, y_ou = us_m.fit_resample(X_o, y_o)
            return X_ou, y_ou

        elif self.method == 'Borderline_SMOTE':
            os_m = BorderlineSMOTE(sampling_strategy=self.sampling_strategy)
            us_m = RandomUnderSampler(sampling_strategy=u_sampling_strategy)
            X_o, y_o = os_m.fit_resample(X=self.X, y=self.y)
            X_ou, y_ou = us_m.fit_resample(X_o, y_o)
            return X_ou, y_ou

        elif self.method == 'ADASYN':
            os_m = ADASYN(sampling_strategy=self.sampling_strategy)
            us_m = RandomUnderSampler(sampling_strategy=u_sampling_strategy)
            X_o, y_o = os_m.fit_resample(X=self.X, y=self.y)
            X_ou, y_ou = us_m.fit_resample(X_o, y_o)
            return X_ou, y_ou

    def visualizing(self, method='O'):
        if method == 'O':
            X, y = self.over_sampling()
        elif method == 'OU':
            X, y = self.over_under_sampling()
        counter = Counter(y)
        title=self.method
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        plt.title(title+'_'+method)
        plt.legend()
        plt.show()

# make datasets
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0)
counter = Counter(y)
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.title('No_oversampling')
plt.legend()
plt.show()


# SMOTE
smt = OversamplingDriver(X, y, method='SMOTE')
smt.visualizing()

# SMOTE with random undersampling of the majority class
smt_RUS = OversamplingDriver(X, y, method='SMOTE', sampling_strategy=0.1)
smt_RUS.visualizing(method='OU')

# BorderLine SMOTE
b_smt = OversamplingDriver(X, y, method='Borderline_SMOTE')
b_smt.visualizing()

# ADASYN
adasyn = OversamplingDriver(X, y, method='ADASYN')
adasyn.visualizing()
