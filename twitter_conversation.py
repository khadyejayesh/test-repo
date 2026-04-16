import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# STEP 1: Load Twitter Data
# -------------------------------
df = pd.read_csv("twitter_edges.csv")

# -------------------------------
# STEP 2: Create Directed Graph
# -------------------------------
G = nx.DiGraph()

for _, row in df.iterrows():
    if row["interaction"] == "mention":
        weight = 1
    elif row["interaction"] == "reply":
        weight = 2
    else:  # retweet
        weight = 3

    G.add_edge(row["Source"], row["Target"], weight=weight)

# -------------------------------
# STEP 3: Network Metrics
# -------------------------------
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
pagerank = nx.pagerank(G, weight="weight")
betweenness = nx.betweenness_centrality(G)

# -------------------------------
# STEP 4: Feature Table
# -------------------------------
features = pd.DataFrame({
    "user": list(G.nodes()),
    "in_degree": [in_degree.get(u, 0) for u in G.nodes()],
    "out_degree": [out_degree.get(u, 0) for u in G.nodes()],
    "pagerank": [pagerank.get(u, 0) for u in G.nodes()],
    "betweenness": [betweenness.get(u, 0) for u in G.nodes()]
})

# -------------------------------
# STEP 5: Machine Learning (Clustering)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    features[["in_degree", "out_degree", "pagerank", "betweenness"]]
)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
features["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# STEP 6: Identify Users
# -------------------------------
most_active = features.sort_values("out_degree", ascending=False).head(5)
most_influential = features.sort_values("pagerank", ascending=False).head(5)

print("\n🔹 Most Active Users:")
print(most_active[["user", "out_degree"]])

print("\n🔹 Most Influential Users:")
print(most_influential[["user", "pagerank"]])

# -------------------------------
# STEP 7: Export for Gephi
# -------------------------------
nx.write_gexf(G, "twitter_network.gexf")
features.to_csv("twitter_user_metrics.csv", index=False)

print("\n✅ Files exported successfully:")
print("1. twitter_network.gexf (for Gephi)")
print("2. twitter_user_metrics.csv (metrics & ML results)")