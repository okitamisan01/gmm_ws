"""
This script fits a Gaussian Mixture Model (GMM) to high-dimensional feature vectors extracted from images.

Steps:
1. Input: Numpy array `features` of shape (n_samples, n_features), e.g. (N, 2048) from CNN.
2. Normalize the feature vectors using StandardScaler.
3. Fit a GMM with a specified number of components (e.g., 10).
4. Predict cluster assignments for all data points.
5. Output: 
    - `cluster_labels`: array of predicted cluster IDs for each sample
    - `gmm`: trained GMM model object

Use scikit-learn's `GaussianMixture` class.

Please generate the code for this pipeline.
"""

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

# features: shape (n_samples, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=0)
gmm.fit(X_scaled)

cluster_labels = gmm.predict(X_scaled)
