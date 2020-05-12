import unittest
import numpy as np

from pca import PCA


class TestPCA(unittest.TestCase):
    def data(self, N):
        A = np.array([[1, 0.5, 0.5], [-1, -1, 2], [1, 2, 1]])
        return np.random.normal(0, 1, size=[N, 3]) @ A

    def test_transform_inverse_transform(self):
        X = self.data(500)
        pca = PCA(X)
        x = pca.transform(X, ndims=3)
        self.assertTrue(np.allclose(np.cov(x.T), np.eye(3), atol=1e-4, rtol=1e-2))
        X_ = pca.inverse_transform(x)
        self.assertTrue(np.allclose(X, X_))

    def test_invariance_to_many_transforms(self):
        X = self.data(500)
        pca = PCA(X)
        x = pca.transform(X, ndims=2)  # error introduced here
        self.assertTrue(np.allclose(np.cov(x.T), np.eye(2), atol=1e-4, rtol=1e-2))
        X1 = pca.inverse_transform(x)
        x1 = pca.transform(X1, ndims=2)
        X2 = pca.inverse_transform(x1)
        self.assertTrue(np.allclose(X1, X2))


TestPCA().test_transform_inverse_transform()
TestPCA().test_invariance_to_many_transforms()
