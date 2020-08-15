# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% pick a criminal code
df_terrorism = df[df["detailcm"] == "TERRORISM"]

df_terrorism["lat"] = df["coord"].apply(lambda val: val[1])
df_terrorism["lon"] = df["coord"].apply(lambda val: val[0])


# %% run hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10

for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_terrorism[["lat", "lon"]])
    silhouette_scores[k] = silhouette_score(df_terrorism[["lat", "lon"]], y)
    labels[k] = y


# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
ax.get_figure().savefig("trend.png", bbox_inches="tight", dpi=400)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Scores")
plt.title("Silhouette Scores vs. Number of Clusters")

# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_terrorism["label"] = labels[best_k]

colors = sns.color_palette("hls", best_k).as_hex()

for row in tqdm(df_terrorism[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m


# %% find reason for stop columns
css = [col for col in df.columns if col.startswith("cs_")]
x = df_terrorism[css] == "YES"


# %% run KMeans clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10
x["lat"] = df_terrorism["lat"]
x["lon"] = df_terrorism["lon"]

for k in tqdm(range(num_city, num_pct, step)):
    km = KMeans(n_clusters=k)
    y = km.fit_predict(x)
    silhouette_scores[k] = silhouette_score(x, y)
    labels[k] = y


# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Scores")
plt.title("Silhouette Scores vs. Number of Clusters")

# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
x["label"] = labels[best_k]

colors = sns.color_palette("hls", best_k).as_hex()

for row in tqdm(x[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m

# %% Clustering for people being searched when crime is terrorism

x = df_terrorism["searched"] == "YES"

# %% run hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

silhouette_scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()
step = 10

for k in tqdm(range(num_city, num_pct, step)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(x)
    silhouette_scores[k] = silhouette_score(x, y)
    labels[k] = y

# %% plot the silhouette_scores agains different numbers of clusters
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),)
ax.get_figure().savefig("trend.png", bbox_inches="tight", dpi=400)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Scores")
plt.title("Silhouette Scores vs. Number of Clusters")

# %% visualize the clustering label on a map
# using the color palette from seaborn
import folium

nyc = (40.730610, -73.935242)
m = folium.Map(location=nyc)

best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])
df_terrorism["label"] = labels[best_k]

colors = sns.color_palette("hls", best_k).as_hex()

for row in tqdm(df_terrorism[["lat", "lon", "label"]].to_dict("records")):
    folium.CircleMarker(
        location=(row["lat"], row["lon"]), radius=1, color=colors[row["label"]]
    ).add_to(m)

m




