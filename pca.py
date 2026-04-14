import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Load dataset
df = sns.load_dataset('titanic')

df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
df = df.dropna()

# Features & target
X = df.drop('survived', axis=1)
y = df['survived']

# Columns
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'embarked', 'pclass']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

print("Explained variance:", pca.explained_variance_ratio_)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Titanic Dataset')
plt.show()