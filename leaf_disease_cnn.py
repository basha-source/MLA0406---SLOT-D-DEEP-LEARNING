import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os

# ---------------- DATASET PATH ----------------

dataset_path = r"F:\Desktop\Asubjects\Deep Learning\lab\leaf_dieases\dataset"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    print("Expected location:", dataset_path)
    exit()

# ---------------- LOAD DATASET ----------------

train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=16,
    class_mode='binary'
)

# ---------------- BUILD CNN ----------------

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

# ---------------- COMPILE ----------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------

model.fit(
    train_generator,
    epochs=5
)

# ---------------- SAVE MODEL ----------------

model.save("leaf_model.h5")

print("Model Saved Successfully")

# ---------------- PREDICT ----------------

image_path = r"F:\Desktop\Asubjects\Deep Learning\lab\leaf_dieases\test_leaf.jpg"

if os.path.exists(image_path):

    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(128,128)
    )

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("Prediction : Healthy")
    else:
        print("Prediction : Diseased")

else:
    print("test_leaf.jpg not found.")