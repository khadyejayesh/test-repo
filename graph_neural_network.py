# Install (run once)
# pip install torch torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Load Cora Dataset
# ----------------------------------
dataset = Planetoid(
    root='data/Planetoid',
    name='Cora',
    transform=NormalizeFeatures()
)

data = dataset[0]
print(data)

# ----------------------------------
# 2. GCN Model for Cora
# ----------------------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# ----------------------------------
# 3. Train Function
# ----------------------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------------
# 4. Test Function
# ----------------------------------
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

# ----------------------------------
# 5. Training Loop
# ----------------------------------
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f"Epoch {epoch}, Loss {loss:.4f}, Test Acc {acc:.4f}")

# ----------------------------------
# 6. Karate Club Dataset
# ----------------------------------
karate_dataset = KarateClub()
karate_data = karate_dataset[0]
print(karate_data)

# ----------------------------------
# 7. GCN for Karate Visualization
# ----------------------------------
class KarateGCN(torch.nn.Module):
    def __init__(self):
        super(KarateGCN, self).__init__()
        self.conv1 = GCNConv(karate_data.num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

karate_model = KarateGCN()
karate_model.eval()

# Get embeddings
embeddings = karate_model(karate_data.x, karate_data.edge_index).detach().numpy()

# ----------------------------------
# 8. Visualization
# ----------------------------------
plt.figure()
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.title("Karate Club Community Structure")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid()
plt.show()