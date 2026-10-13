import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = torch.load("/storage/adhin/latent_space/tokens_train.pt", weights_only=False,map_location="cpu")

token = data["tokens"].float().mean(dim=(2,3)).numpy() #tokens is a string not array so convert
doys = data["doys"].numpy()
hours = data["hours"].numpy()

token_scaled = StandardScaler().fit_transform(token)
kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)
kmeans.fit(token_scaled)

df = pd.DataFrame(token[:,:10], columns=[f"ch{i}" for i in range(10)])
df["doy"] = doys
df["hour"] = hours
df["timestep"] = np.arange(len(doys))
df["Segment K-means"] = kmeans.labels_
df["Segment"] = "cluster " + df["Segment K-means"].astype(str)

plt.figure(figsize=(20, 10))
sns.scatterplot(x=df["doy"], y=df["ch0"], hue=df["Segment"],
                palette=["g", "r", "c", "m", "orange"], s=8, alpha=0.4)
plt.xlabel("Day of the Year"); plt.ylabel("Token Channel 0")
plt.title("K-means (no PCA) clusters, raw channel 0 vs day of year")
plt.savefig("kmeans_doy_vs_token.png", dpi=150)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=df["ch0"], y=df["ch1"], hue=df["Segment"],
                palette=["g", "r", "c", "m", "orange"], s=8, alpha=0.4)
plt.xlabel("Token Channel 0"); plt.ylabel("Token Channel 1")
plt.title("K-means (no PCA), raw values")
plt.savefig("kmeans_ch0_ch1.png", dpi=150)
