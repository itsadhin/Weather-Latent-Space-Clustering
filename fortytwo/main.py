import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = torch.load("/storage/adhin/latent_space/tokens_train.pt", weights_only=False,map_location="cpu")

token = data["tokens"].float().mean(dim=(2,3)).numpy() #tokens is a string not array so convert
doys = data["doys"].numpy()
hours = data["hours"].numpy()

#-------------------------
# Trying to Visualize data
#-------------------------
print(token)
#print(len(token))
#print(len(doys))
plt.figure(figsize=(20,10))
plt.scatter(np.arange(len(doys)),token[:,0],lw=0.3)
plt.xlabel("Timestep")
plt.ylabel("Token Channel 0")
plt.title("Timeseries")
plt.savefig("timeseries.png")

plt.figure(figsize=(20,10))
plt.scatter(doys,token[:,0],alpha=0.3)
plt.xlabel("Day of the Year")
plt.ylabel("Token Channel 0")
plt.savefig("doy_vs_token.png")

plt.figure(figsize=(20,10))
plt.scatter(doys,token.mean(1),s=1,alpha=0.3)
plt.xlabel("Day of Year")
plt.ylabel("Mean over 1024 Channel")
plt.savefig("Mean_over.png")

#----------------
# Applying PCA
#----------------
scaler = StandardScaler()
pca = PCA()
token_transformed = scaler.fit_transform(token)
pca.fit(token_transformed)                 #Standardizing data using PCA

#print(pca.explained_variance_ratio_)
plt.figure(figsize=(12,9))
plt.plot(np.arange(len(pca.explained_variance_ratio_),pca.explained_variance_ratio_.cumcum(), marker="*", linestyle="--"))
plt.title("Explained Variance by component")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Vectors")
plt.savefig("explained_var.png")

