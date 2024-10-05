import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19
from tensorflow.keras.models import Sequential
from datetime import datetime
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop  

time = datetime.now().strftime("%y-%m-%dT%H-%M")

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

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted') 

    plt.ylabel('Actual')

    # Save the image
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.clf()
    
def alexnet(num_classes):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust the number of classes as needed
    
    return model


num_images = -1
X, y = load_images("/home/datasets/spark22-dataset",num_images)
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)
X= X/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = 11

resnet50 = Sequential([
        ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation='softmax')])

vgg16_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
vgg16_base.trainable= False
vgg16_model = Sequential([
        vgg16_base,
        Flatten(),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')])

vgg19_base = VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=(224, 224, 3))
vgg19_base.trainable= False
vgg19_model = Sequential([
        vgg19_base,
        Flatten(),
        Dense(4096,activation="sigmoid"),
        Dropout(0.5),
        Dense(4096,activation="sigmoid"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")])

googleNet = Sequential([
        InceptionV3(include_top=False, input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation="softmax"),
        ])

alex = alexnet(num_classes)

models = {"alexnet":alex,
          "resnet50": resnet50,
          "vgg16":vgg16_model,
          "vgg19":vgg19_model,
          "googlenet":googleNet,
          }

for name, model in models.items():
    #optim = Adam()#learning_rate=0.0005, beta_1=0.9999, beta_2=0.999, epsilon=1e-8)
    optim = RMSprop()
    if False and name =="vgg16":
        x = vgg16.preprocess_input(X_train)
        x_test= vgg16.preprocess_input(X_test)
    
    elif False and name =="vgg19":
        x = vgg16.preprocess_input(X_train)
        x_test= vgg16.preprocess_input(X_test)
    else:
        x= X_train
        x_test = X_test
         
    #optim  = Nadam(learning_rate=0.001)
    model.compile(optimizer=optim,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    model.fit(x, y_train, epochs=10)
    
    model.save(f"{name}_{time}.keras") 

    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred = tf.argmax(y_pred,axis=-1)
    y_t = tf.argmax(y_test,axis=-1)
    
    print(y_pred.shape)
    print(y_t.shape)
    results = classification_report(y_true=y_t, y_pred=y_pred)
       
    with open(f"{name}_{time}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(results))
    save_conf_mat(y_t, y_pred, name=f"{name}_{time}")