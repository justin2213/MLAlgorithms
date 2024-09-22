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
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Use color images
                image = cv2.resize(image, None, fx=0.0625, fy=0.0625)  # Resize the image
                images.append(np.reshape(image, (-1,)))  # Flatten the image
                print(image.shape)

                # Use the folder name as the label
                labels.append(root.split("/")[-1])
    
    return np.vstack(images), labels

def save_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted') 
    plt.ylabel('Actual')
    # Save the image
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.clf()

# Load images and labels
num_images = 1000  # Use more images for training if possible
X, y = load_images("/home/datasets/spark22-dataset", num_images)

# Convert string labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define the pipeline without PCA
pipeline = make_pipeline(
    StandardScaler(),  # Normalize the data
    XGBClassifier(random_state=42),  # XGBoost classifier
    verbose=True
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
with open(f"xgb_metrics.txt", "w") as f:
    f.write(str(metrics))

save_conf_mat(y_test, y_pred, name="xgb")

# Optionally, print out the classification report
print(metrics)
