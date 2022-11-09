import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
from sklearn.model_selection import train_test_split

class KNN:
    """KNNModel"""

    def __init__(self, n=3, dipole_dist_weight=5, dipolarophile_dist_weight=1, product_dist_weight=1, **kwargs):
        self.n = n
        self.dipole_dist_weight = dipole_dist_weight
        self.dipolarophile_dist_weight = dipolarophile_dist_weight
        self.product_dist_weight = product_dist_weight

    def cosine_dist(self, train_objs, test_objs):
        """compute cosine_dist"""
        numerator = np.einsum('ij,kj->ik', train_objs, test_objs)

        norm = lambda x: (x**2).sum(-1)**(0.5)

        denominator = norm(train_objs)[:, None] * norm(test_objs)[None, :]
        denominator[denominator == 0] = 1e-12
        cos_dist = 1 - numerator / denominator

        return cos_dist

    def fit(self, data, train_vals) -> None:
        self.train_dipoles = data[0]
        self.train_dipolarophiles = data[1]
        self.train_products = data[2]
        self.train_vals = train_vals

    def predict(self, data) -> np.ndarray:
        # Compute test dists
        test_dipoles, test_dipolarophiles, test_products = data
        test_dipole_dists = self.cosine_dist(self.train_dipoles, test_dipoles)
        test_dipolarophile_dists = self.cosine_dist(self.train_dipolarophiles, test_dipolarophiles)
        test_product_dists = self.cosine_dist(self.train_products, test_products)
        total_dists = (self.dipole_dist_weight * test_dipole_dists +
                       self.dipolarophile_dist_weight * test_dipolarophile_dists + self.product_dist_weight * test_product_dists)

        smallest_dists = np.argsort(total_dists, 0)

        top_n = smallest_dists[:self.n, :]
        ref_vals = self.train_vals[top_n]
        mean_preds = np.mean(ref_vals, 0)
        return mean_preds
