import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
sns.set()
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import normalized_mutual_info_score

data = torch.load("/storage/adhin/latent_space/tokens_train.pt", weights_only=False,map_location="cpu")
token = data["tokens"].float().mean(dim=(2,3)).numpy()
token_scaled = StandardScaler().fit_transform(token)

pca = PCA(n_components=0.90)
pca.fit(token_scaled)
scores_pca = pca.transform(token_scaled)
u = umap.UMAP(n_components=10, random_state=42).fit_transform(scores_pca)
for k in range(2, 20):
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(u)
    print(k,
          round(silhouette_score(u, lab, sample_size=10000, random_state=42), 3),
          round(calinski_harabasz_score(u, lab)),
          round(davies_bouldin_score(u, lab), 3))

kmeans = KMeans(n_clusters=14,init="k-means++",random_state=42)
tsne = TSNE(n_components=3,random_state=0)
t = tsne.fit_predict(scores_pca)
labels = kmeans.fit_predict(t)

x = t[:,0]
y = t[:,1]
z = t[:,2]
ts = pd.to_datetime(pd.Series(np.asarray(data["timestamps"]).reshape(-1)))
y_true = (ts.dt.month.to_numpy() % 12) // 3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=labels, cmap="Set2", s=2, alpha=0.5)
print("Silhouette Score: ", silhouette_score(t, labels))
print("Adjusted Random Index: ", adjusted_rand_score(y_true,labels))
print("Random Index: ", rand_score(y_true,labels))
ax.set_title("KMeans with PCA 3D + tSNE \nEstimated number of clusters: 14")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")
plt.legend()
plt.show()

