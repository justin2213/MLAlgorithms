import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imgbeddings import imgbeddings

# Load images and convert to embeddings
def load_images(dir, limit=100):
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                paths.append(image_path)

    ibed = imgbeddings()
    emb = ibed.to_embeddings(paths)
    
    return np.vstack(emb)

# Plot clusters after dimensionality reduction
def plot_clusters(X, labels, title):
    # Reduce to 2 dimensions for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot the 2D data
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.clf()

# Load images and create embeddings
X = load_images("/home/datasets/spark22-dataset", 1000)

# Set number of clusters
k = 11

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Plot the clustered data
plot_clusters(X, labels, title="K Means Clustering with PCA")
