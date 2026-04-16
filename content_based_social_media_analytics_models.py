# Practical 5
# Import required libraries
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman

# -----------------------------
# 1. Create Graph
# -----------------------------
G = nx.Graph()

# Define edges (user connections)
edges = [
    ('User1', 'User2'),
    ('User2', 'User3'),
    ('User3', 'User4'),
    ('User4', 'User1'),
    ('User2', 'User4'),
    ('User5', 'User6')
]

# Add edges to graph
G.add_edges_from(edges)

# -----------------------------
# 2. Community Detection (Girvan-Newman)
# -----------------------------
communities = girvan_newman(G)
first_level_communities = next(communities)

print("Communities:")
for i, comm in enumerate(first_level_communities, 1):
    print(f"Community {i}: {list(comm)}")

# -----------------------------
# 3. Influence Analysis
# -----------------------------
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print("\nDegree Centrality:")
for node, value in degree_centrality.items():
    print(f"{node}: {round(value, 3)}")

print("\nBetweenness Centrality:")
for node, value in betweenness_centrality.items():
    print(f"{node}: {round(value, 3)}")

# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(6, 6))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    with_labels=True,
    node_size=2000,
    font_size=10
)

plt.title("Social Network Graph")
plt.show()