import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load sample customer data
# If you have your own CSV: pd.read_csv('your_file.csv')

df = pd.read_csv('Mall_Customers.csv')
print(df.head())
# We'll use Annual Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Use Elbow Method to find optimal clusters
inertia = []
for k in range(1, 11):
   km = KMeans(n_clusters=k, random_state=42)
   km.fit(X_scaled)
   inertia.append(km.inertia_)




# Plot elbow
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
# Choose K=5 based on elbow curve
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
