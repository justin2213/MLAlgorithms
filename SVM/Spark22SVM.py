from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os 
import cv2
import numpy as np


def load_images(dir,limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                image = cv2.resize(image, None, fx=0.0625, fy=0.0625)
                print(image.shape)

                images.append(np.reshape(image,(-1,)))
                
                labels.append(root.split("/")[-1])
    
    return np.vstack(images), labels


kernals = ['rbf', 'linear', 'poly',  'sigmoid']
num_images = 1000
X, y = load_images("/home/datasets/spark22-dataset",num_images)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svms    = {}
results = {}
metrics = {}
for kernal in kernals:
    print(kernal)
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal,verbose=2))
    svms[kernal] = svms[kernal].fit(X_train,y_train)
    results[kernal]= svms[kernal].predict(X_test)
    metrics[kernal] = classification_report(y_true=y_test,y_pred=results[kernal])
   
    with open(f"{kernal}_metric.txt", "w") as f:
            # Write some text to the file
            f.write(str(metrics[kernal]))


    cm = confusion_matrix(y_test, results[kernal])

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {kernal}')
    plt.xlabel('Predicted') 

    plt.ylabel('Actual')

    # Save the image
    plt.savefig(f'{kernal}_confusion_matrix.png')
    plt.clf()
