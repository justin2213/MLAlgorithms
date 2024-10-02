import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential


def load_images(dir,limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224,224))
                images.append(image) 
                labels.append(root.split("/")[-1])
    
    return np.array(images), labels

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
num_images = 10000  
X, y = load_images("/home/datasets/spark22-dataset", num_images)

y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)

# Convert string labels to numeric values
label_encoder = OneHotEncoder(sparse_output=False)
y = label_encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of image classes
num_classes = 11

model = Sequential([
    ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    Flatten(),
    Dropout(0.5),  # Add dropout with 50% dropout rate
    Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Fit the model on the training data
model.fit(X_train, y_train, epochs=10)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels
y_true = np.argmax(y_test, axis=1)

# Save the metrics and confusion matrix
metrics = classification_report(y_true, y_pred_classes)

with open("resnet_metrics.txt", "w") as f:
    f.write(str(metrics))

save_conf_mat(y_true, y_pred_classes, name="resnet")
