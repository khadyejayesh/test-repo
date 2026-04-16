import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# -------------------------------
# 1. Les Misérables Dataset
# -------------------------------
G = nx.les_miserables_graph()

# 1. Persons-by-Persons (One-Mode)
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
plt.title("Les Misérables – Persons-by-Persons Sociogram (One-Mode)")
plt.show()

# ---------------------------------
# 2. Community-by-Community Network
# ---------------------------------
communities = list(greedy_modularity_communities(G))
C = nx.Graph()

for i, c1 in enumerate(communities):
    C.add_node(i)
    for j, c2 in enumerate(communities):
        if i < j:
            weight = sum(1 for u in c1 for v in c2 if G.has_edge(u, v))
            if weight > 0:
                C.add_edge(i, j, weight=weight)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(C)
nx.draw(C, pos, with_labels=True, node_size=1000)
plt.title("Les Misérables – Community-by-Community Network")
plt.show()

# -------------------------------
# 3. Two-Mode Network
# -------------------------------
B = nx.Graph()
characters = list(G.nodes())[:8]
chapters = ["Chapter-1", "Chapter-2", "Chapter-3"]

B.add_nodes_from(characters, bipartite=0)
B.add_nodes_from(chapters, bipartite=1)

for char in characters:
    B.add_edge(char, "Chapter-1")

plt.figure(figsize=(9, 6))
pos = nx.spring_layout(B)
nx.draw(B, pos, with_labels=True, node_size=500)
plt.title("Les Misérables – Two-Mode Network")
plt.show()

# -------------------------------
# Airlines Network Dataset
# -------------------------------
G_air = nx.Graph()
routes = [
    ("Delhi", "Mumbai"),
    ("Mumbai", "Bangalore"),
    ("Delhi", "Bangalore"),
    ("Mumbai", "Chennai")
]

G_air.add_edges_from(routes)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_air)
nx.draw(G_air, pos, with_labels=True, node_size=800)
plt.title("Airlines – Airport-by-Airport Network")
plt.show()

# Community Network
C_air = nx.Graph()
alliances = ["Star Alliance", "OneWorld", "SkyTeam"]

C_air.add_nodes_from(alliances)
C_air.add_edge("Star Alliance", "OneWorld")
C_air.add_edge("Star Alliance", "SkyTeam")

plt.figure(figsize=(6, 5))
pos = nx.spring_layout(C_air)
nx.draw(C_air, pos, with_labels=True, node_size=1200)
plt.title("Airlines – Community Network")
plt.show()

# Two-Mode Network
B_air = nx.Graph()
airlines = ["IndiGo", "Air India"]
airports = ["Delhi", "Mumbai", "Bangalore"]

B_air.add_nodes_from(airlines, bipartite=0)
B_air.add_nodes_from(airports, bipartite=1)

B_air.add_edge("IndiGo", "Delhi")
B_air.add_edge("IndiGo", "Mumbai")
B_air.add_edge("Air India", "Bangalore")

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(B_air)
nx.draw(B_air, pos, with_labels=True, node_size=600)
plt.title("Airlines – Two-Mode Network")
plt.show()

# -------------------------------
# Internet Core Network Dataset
# -------------------------------
G_as = nx.Graph()
edges = [
    ("AS123", "AS456"),
    ("AS456", "AS789"),
    ("AS123", "AS789")
]

G_as.add_edges_from(edges)

plt.figure(figsize=(7, 5))
pos = nx.spring_layout(G_as)
nx.draw(G_as, pos, with_labels=True, node_size=800)
plt.title("Internet Core – AS-to-AS Network")
plt.show()

# Community Network
C_as = nx.Graph()
tiers = ["Tier-1 ISP", "Tier-2 ISP", "Regional ISP"]

C_as.add_nodes_from(tiers)
C_as.add_edge("Tier-1 ISP", "Tier-2 ISP")
C_as.add_edge("Tier-2 ISP", "Regional ISP")

plt.figure(figsize=(6, 5))
pos = nx.spring_layout(C_as)
nx.draw(C_as, pos, with_labels=True, node_size=1200)
plt.title("Internet Core – Community Network")
plt.show()

# Two-Mode Network
B_as = nx.Graph()
AS_nodes = ["AS123", "AS456"]
countries = ["India", "USA"]

B_as.add_nodes_from(AS_nodes, bipartite=0)
B_as.add_nodes_from(countries, bipartite=1)

B_as.add_edge("AS123", "India")
B_as.add_edge("AS456", "USA")

plt.figure(figsize=(7, 5))
pos = nx.spring_layout(B_as)
nx.draw(B_as, pos, with_labels=True, node_size=600)
plt.title("Internet Core – Two-Mode Network")
plt.show()