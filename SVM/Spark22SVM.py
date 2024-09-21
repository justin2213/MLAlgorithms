from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
import seaborn as sns



def load_image_files(container_path, dimension=(1024, 1024), max_images_per_class = 500):
    """
    Load a limited number of image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which images are adjusted to
    max_images_per_class : int
        Maximum number of images to load per class
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [folder.name for folder in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    
    for i, direc in enumerate(folders):
        count = 0  # Counter to limit number of images per class
        for file in direc.iterdir():
            if count >= max_images_per_class:
                break  # Stop if we've reached the limit for this class
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
            count += 1  # Increment the counter
            print(count, i)
        
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

param_grids = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'coef0': [0, 1, 10], 'kernel': ['sigmoid']}
]
image_dataset = load_image_files('/home/datasets/spark22-dataset/train')
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

for grid in param_grids:
    print(grid)
    svc = svm.SVC(random_state=42)
    
    # Perform GridSearchCV with the kernel
    clf = GridSearchCV(svc, grid, cv=5)
    clf.fit(X_train, y_train)
    
    # Predict the labels for the test set
    y_pred = clf.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save confusion matrix as a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=image_dataset.target_names,
                yticklabels=image_dataset.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot to a file
    plt.savefig(f'confusion_matrix_{grid["kernel"][0]}.png')
    plt.close()  # Close the figure after saving

    # Save classification report to a text file
    report = classification_report(y_test, y_pred, target_names=image_dataset.target_names)
    with open(f'classification_report_{grid["kernel"][0]}.txt', 'w') as f:
        f.write(report)

    print(f"Results saved for kernel: {grid['kernel'][0]}")

