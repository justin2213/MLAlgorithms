from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
import numpy as np
from xgboost import XGBClassifier
import os

def load_images(dir, limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (256, 256))  # Resize to 256x256
                images.append(image.flatten())  # Flatten the image
                labels.append(root.split("/")[-1])  # Use folder name as label
    
    return np.array(images), labels  # Return as a NumPy array

def save_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.clf()

# Load images and labels
num_images = -1
X, y = load_images("/home/datasets/spark22-dataset", num_images)

# Convert string labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Create a pipeline with StandardScaler, PCA, and XGBoost classifier
pipeline = make_pipeline(
    StandardScaler(),  # Normalize the data
    PCA(n_components=50),  # Reduce to 50 principal components
    XGBClassifier(random_state=42)  # XGBoost classifier
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
metrics = classification_report(y_test, y_pred)

# Save the metrics and confusion matrix
with open(f"xgb_pca_metrics.txt", "w") as f:
    f.write(str(metrics))

save_conf_mat(y_test, y_pred, name="xgb_pca")

# Optionally, print out the classification report
print(metrics)
