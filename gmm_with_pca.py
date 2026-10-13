import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

data = torch.load("/storage/adhin/latent_space/tokens_train.pt",map_location='cpu',weights_only=False)

token = data["tokens"].float().mean(dim=(2,3)).numpy() #tokens is a string not array so convert
doys = data["doys"].numpy()
hours = data["hours"].numpy()

scaler = StandardScaler()
pca = PCA()
token_transformed = scaler.fit_transform(token)
pca.fit(token_transformed)
pca = PCA(n_components = 62)
pca.fit(token_transformed)     #Standardizing data using PCA
scores_pca = pca.transform(token_transformed)

gmm = GaussianMixture(n_components=5,covariance_type="diag",n_init=3,random_state=42)
gmm.fit(scores_pca)
labels = gmm.predict(scores_pca)
probs = gmm.predict_proba(scores_pca)

df = pd.DataFrame(scores_pca[:, :10], columns=[f"Component {i+1}" for i in range(10)])
df["doy"] = doys
df["hour"] = hours
df["timestep"] = np.arange(len(doys))
df["Segment GMM PCA"] = labels
df["Segment"] = "cluster " + df["Segment GMM PCA"].astype(str)
df["confidence"] = probs.max(1)

order = [f"cluster {i}" for i in range(5)]
pal = ["g", "r", "c", "m", "orange"]

order = [f"cluster {i}" for i in range(5)]
pal = ["g", "r", "c", "m", "orange"]

plt.figure(figsize=(10, 8))
sns.scatterplot(x=df["Component 1"], y=df["Component 2"],
                hue=df["Segment"], hue_order=order, palette=pal,
                s=6, alpha=0.4)
plt.xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("GMM clusters: PC1 vs PC2")
plt.savefig("gmm_pca1_pca2.png", dpi=150)

plt.figure(figsize=(10, 8))
plt.scatter(df["Component 1"], df["Component 2"], c=df["doy"], s=3, cmap="twilight")
plt.colorbar(label="day of year")
plt.xlabel("Component 1"); plt.ylabel("Component 2")
plt.title("PC1 vs PC2 colored by day of year")
plt.savefig("pca_by_doy_gmm.png", dpi=150)

