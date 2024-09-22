import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os 
import numpy as np
from PIL import Image

from imgbeddings import imgbeddings

def load_images(dir,limit=100):
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                paths.append(image_path)

    ibed = imgbeddings()

    emb = ibed.to_embeddings(paths)
    
    return np.vstack(emb)

def plot_clusters(images, labels, n_clusters):
    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        cluster_images = images[labels == i]
        plt.subplot(1, n_clusters, i + 1)
        plt.title(f'Cluster {i}')
        plt.axis('off')
        
        # Show the first image of the cluster
        if len(cluster_images) > 0:
            plt.imshow(cluster_images[0].reshape(64, 64), cmap='gray')
    plt.tight_layout()
    plt.show()

# Load images
X = load_images("/home/datasets/spark22-dataset", 1000)

k = 5

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
plot_clusters(X, labels, k)

