import networkx as nx
import matplotlib.pyplot as plt

# Fix import (important)
try:
    import community.community_louvain as community_louvain
except ImportError:
    print("❌ ERROR: python-louvain not installed")
    print("👉 Run: pip install python-louvain")
    exit()

print("✅ Program started")

# ------------------------------
# Step 1: Create Graph
# ------------------------------
G = nx.Graph()

users = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
G.add_nodes_from(users)

edges = [
    ("A", "B"), ("A", "C"), ("B", "C"),
    ("D", "E"), ("E", "F"), ("D", "F"),
    ("G", "H"), ("H", "I"), ("G", "I"),
    ("C", "D"),
    ("F", "G")
]
G.add_edges_from(edges)

print("✅ Graph created")

# ------------------------------
# Step 2: Community Detection
# ------------------------------
partition = community_louvain.best_partition(G)

print("\n--- Community Detection ---")
for node in sorted(partition):
    print(f"User {node} -> Community {partition[node]}")

# ------------------------------
# Step 3: Influence Analysis
# ------------------------------
print("\n--- Influence Analysis ---")

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

print("\nDegree Centrality:")
for node in sorted(degree_centrality):
    print(f"{node}: {degree_centrality[node]:.3f}")

print("\nBetweenness Centrality:")
for node in sorted(betweenness_centrality):
    print(f"{node}: {betweenness_centrality[node]:.3f}")

print("\nPageRank:")
for node in sorted(pagerank):
    print(f"{node}: {pagerank[node]:.3f}")

# ------------------------------
# Step 4: Visualization
# ------------------------------
pos = nx.spring_layout(G, seed=42)

colors = [partition[node] for node in G.nodes()]

plt.figure(figsize=(8, 6))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=colors,
    cmap=plt.cm.Set3,
    node_size=1000,
    font_size=10
)

plt.title("Social Network - Community Structure")

# Save instead of only show (important fix)
plt.savefig("graph.png")
print("\n✅ Graph saved as graph.png")

plt.show()