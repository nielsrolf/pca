import numpy as np
from matplotlib import pyplot as plt


class PCA():
    """
    X, x : (N, d), assume zero mean
    U: (N, N)
    S: (N, d)
    Vh: (d, d)
    U S Vh = X

    # transform:
    x  := X Vh.T S^-1 /N = U S Vh Vh.T (S/sqrt(N))^(-1)
        = U S S^(-1) * sqrt(N)
        = U * sqrt(N)
    cov(x.T) = 1/N x.T x = 1/N sqrt(N) U.T U sqrt(N) = I_d
        => x is the whitened projection of X with identity covariance
    X = x (S/sqrt(N)) Vh
    """

    def __init__(self, data):
        self.shape = list(data.shape[1:])
        data = self.flatten(data)
        self.mean = data.mean(axis=0)
        self.u, s, self.vh = np.linalg.svd(data - self.mean)
        self.s = s / np.sqrt(len(data))

    @staticmethod
    def flatten(data):
        N = len(data)
        return data.reshape((N, -1))

    def to_original_shape(self, data):
        N = len(data)
        return data.reshape([N] + self.shape)

    def plot_eigenvalues(self):
        plt.plot(self.s)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.show()

    def get_eigenvectors(self, ndims):
        return self.vh[:ndims].reshape([ndims] + self.shape)

    def decorrelate(self, data, ndims=2):
        data = self.flatten(data)
        return (data - self.mean) @ (self.vh[:ndims]).T

    def transform(self, data, ndims=2):
        """Project onto n-dimensional subspace"""
        data = self.flatten(data)
        return (data - self.mean) @ (self.vh[:ndims]).T / self.s[:ndims]

    def inverse_transform(self, p):
        """Project from low dimensional projection into original space"""
        ndims = p.shape[1]
        return self.to_original_shape(p * self.s[:ndims] @ self.vh[:ndims] + self.mean)

    def plot_reconstruction_error_over_ndims(self, data):
        error = []
        NDIMS = list(range(1, self.flatten(data).shape[1], 20))
        for ndims in NDIMS:
            x = self.transform(data, ndims)
            X = self.inverse_transform(x)
            error += [np.mean(np.square(data - X))]
        plt.plot(NDIMS, error)
        plt.xlabel("ndims")
        plt.ylabel("MSE(X, X_)")
        plt.show()
