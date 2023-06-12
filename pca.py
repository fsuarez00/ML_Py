import numpy as np


class PCA:

	def __init__(self, k):
		self.k = k
		self.components = None
		self.mean = None

	def fit(self, X):
		# mean
		self.mean = np.mean(X, axis=0)
		X -= self.mean

		# covariance
		cov = np.cov(X.T)

		# eigenvectors, eigenvalues
		eigenvalues, eigenvectors = np.linalg.eig(cov)

		# sort eigenvectors
		eigenvectors = eigenvectors.T
		idxs = np.argsort(eigenvalues)[::-1]  # decreasing order
		eigenvalues = eigenvalues[idxs]
		eigenvectors = eigenvectors[idxs]

		# store first k eigenvectors
		self.components = eigenvectors[0:self.k]

	def transform(self, X):
		# project data
		X -= self.mean
		return np.dot(X, self.components.T)
