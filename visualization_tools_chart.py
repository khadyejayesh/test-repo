import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------------------------
# Load data (change path if needed)
# ----------------------------------
data = pd.read_csv("social_network.csv")

# Create directed graph
G = nx.from_pandas_edgelist(
    data,
    source='Source',
    target='Target',
    create_using=nx.DiGraph()
)

# ----------------------------------
# Independent Cascade Model
# ----------------------------------
def independent_cascade(G, start_node, prob=0.4):
    active = set([start_node])
    newly_active = set([start_node])

    while newly_active:
        next_active = set()

        for node in newly_active:
            for neighbor in G.neighbors(node):
                if neighbor not in active:
                    if random.random() < prob:
                        next_active.add(neighbor)

        newly_active = next_active
        active.update(newly_active)

    return active

# ----------------------------------
# Run simulation
# ----------------------------------
start = "A"
activated_nodes = independent_cascade(G, start)

print("Nodes influenced:", activated_nodes)
print("Total influenced:", len(activated_nodes))

# ----------------------------------
# Visualization
# ----------------------------------
plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G, seed=42)

node_colors = []
for node in G.nodes():
    if node in activated_nodes:
        node_colors.append("red")
    else:
        node_colors.append("lightblue")

nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=2000,
    arrows=True
)

plt.title("Information Diffusion Simulation")
plt.show()