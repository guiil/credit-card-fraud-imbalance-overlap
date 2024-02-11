import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class RValue(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=7, theta=3):
        self.n_neighbors = n_neighbors
        self.theta = theta

    def fit(self, x, y=None):

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        neigh.fit(x)
        total_samples = len(x)

        total_overlapping_samples = 0

        # initiate Metrics dict
        metrics_dict = {
            'r(f)': 0,
        }
        for c in y.unique():
            metrics_dict[f'r(C{c})'] = 0

        for idx, sample in enumerate(x):

            sample_class = y.iloc[idx]

            n_nearests_neighbors = np.delete(
                neigh.kneighbors([sample])[1][0],
                0
            )

            sample_neighbors_classes = []
            for neighbor in n_nearests_neighbors:
                sample_neighbors_classes.append(y.iloc[neighbor])

            set_knn_diff_classes = list(
                filter(
                    (sample_class).__ne__,
                    sample_neighbors_classes
                )
            )

            # Formula =
            # 'knn(Pim, U-S(Ci)) = set_knn_diff_classes
            # '|knn(Pim, U-S(Ci))| = len(set_knn_diff_classes)
            overlap = 1 if (len(set_knn_diff_classes) - self.theta > 0) else 0

            metrics_dict['r(f)'] += 1 if overlap == 1 else 0
            metrics_dict[f'r(C{sample_class})'] += 1 if overlap == 1 else 0

            total_overlapping_samples += 1 if overlap == 1 else 0

        metrics_dict['total_overlapping_samples'] = total_overlapping_samples
        metrics_dict['R(f)'] = total_overlapping_samples/total_samples

        for c in y.unique():
            metrics_dict[f'R(C{c})'] = (
                metrics_dict[f'r(C{c})'] /
                len(y[y == c])
            )

        IR = len(y[y == 0])/len(y[y == 1])
        metrics_dict['IR'] = IR

        r_aug = (
            (1/(IR + 1)) *
            (metrics_dict['R(C0)'] + IR * metrics_dict['R(C1)'])
        )
        metrics_dict['R_aug(f)'] = r_aug

        print(metrics_dict)

        self.r_values = metrics_dict
        return self

    def transform(self, x, y=None):
        return x
