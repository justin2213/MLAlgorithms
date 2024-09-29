from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
import numpy as np
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load images function
def load_images(dir, limit=100):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (224, 224))  # Required size for ResNet
                image = preprocess_input(image)
                image = np.expand_dims(image, axis=0)
                features = model.predict(image)
                images.append(features.flatten())
                labels.append(root.split("/")[-1])
    return np.array(images), labels

# Save confusion matrix function
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
num_images = 10000 # Use more images for training if possible
X, y = load_images("/home/datasets/spark22-dataset", num_images)

# Convert string labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define the pipeline with PCA
pipeline = make_pipeline(
    StandardScaler(),  # Normalize the data
    XGBClassifier(
        random_state=42,
        colsample_bytree=0.9, # 0.9 seems to work best
        max_depth=3, # 3 seems to work best
        learning_rate=0.2, # 0.2 seems to work best
        n_estimators=300, # 300 works best
        subsample=0.8, # 0.8 works best
        min_child_weight = 3, # 3 works best 
        reg_alpha = 0.01
    ), # XGBoost classifier
    verbose=True
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Make predictions on the test set using the best model
y_pred = pipeline.predict(X_test)

# Evaluate the model
metrics = classification_report(y_test, y_pred)

# Save the metrics and confusion matrix
with open(f"xgb_metrics.txt", "w") as f:
    f.write(str(metrics))

save_conf_mat(y_test, y_pred, name="xgb")

# Optionally, print out the classification report
print(metrics)
