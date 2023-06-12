from sklearn.datasets import make_blobs
import numpy as np
from kmeans import KMeans

X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(k=clusters, max_iters=150, plot_steps=False)
y_pred = k.predict(X)

k.plot()
