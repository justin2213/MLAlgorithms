from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from scipy.stats import mode
import numpy as np
from sklearn.metrics import classification_report

# Load image dataset
def load_image_files(container_path, dimension=(64, 64), max_images_per_class=1000):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [folder.name for folder in folders]
    descr = "An image classification dataset"
    images, flat_data, target = [], [], []
    
    for i, direc in enumerate(folders):
        count = 0
        for file in direc.iterdir():
            if count >= max_images_per_class:
                break
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
            count += 1
            print(count, i)
            
    return Bunch(data=np.array(flat_data),
                 target=np.array(target),
                 target_names=categories,
                 images=np.array(images),
                 DESCR=descr)

def map_clusters_to_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        # Assign the most common true label in the cluster to the cluster
        labels[mask] = mode(y_true[mask])[0][0]
    return labels

# Load dataset
image_dataset = load_image_files('/path/to/dataset')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

# Initialize K-Means model
kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=42)

# Fit K-Means model to training data
kmeans.fit(X_train)

# Predict clusters for training and test sets
y_train_pred = kmeans.predict(X_train)
y_test_pred = kmeans.predict(X_test)

# Evaluate results
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_test_mapped = map_clusters_to_labels(y_test, y_test_pred)

# Generate classification report
class_report = classification_report(y_test, y_test_mapped)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_mapped)



# Write the confusion matrix to a text file
conf_matrix_file_path = Path("confusion_matrix.txt")
with conf_matrix_file_path.open("w") as conf_matrix_file:
    conf_matrix_file.write("Confusion Matrix:\n")
    np.savetxt(conf_matrix_file, conf_matrix, fmt="%d")

# Write the classification report to a text file
report_file_path = Path("classification_report.txt")
with report_file_path.open("w") as report_file:
    report_file.write("Classification report for K-Means:\n")
    report_file.write(class_report)

