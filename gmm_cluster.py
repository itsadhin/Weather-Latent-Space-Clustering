import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
sns.set()
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

data = torch.load("/storage/adhin/latent_space/tokens_train.pt", map_location='cpu',weights_only=False)
token = data["tokens"].float().mean(dim=(2,3)).numpy()
doys = data["doys"].numpy()
hours = data["hours"].numpy()

token_scaled = StandardScaler().fit_transform(token)
gmm = GaussianMixture(n_components=5, covariance_type='diag',n_init=3, random_state=42)
gmm.fit(token_scaled)
labels = gmm.predict(token_scaled)
probs = gmm.predict_proba(token_scaled)

df = pd.DataFrame(token[:, :10], columns=[f"ch{i}" for i in range(10)])  #10 channels
df["doy"] = doys
df["hour"] = hours
df["timestep"] = np.arange(len(doys))
df["Segment GMM"] = labels
df["Segment"] = "cluster " + df["Segment GMM"].astype(str)
df["confidence"] = probs.max(1)

order = [f"cluster {i}" for i in range(5)]

plt.figure(figsize=(20, 10))
sns.scatterplot(x=df["doy"], y=df["ch0"], hue=df["Segment"], hue_order=order,
                palette=["g", "r", "c", "m", "orange"], s=8, alpha=0.4)
plt.xlabel("Day of the Year"); plt.ylabel("Token Channel 0")
plt.title("GMM (no PCA) clusters, raw channel 0 vs day of year")
plt.savefig("gmm_doy_vs_token.png", dpi=150)
plt.close()
