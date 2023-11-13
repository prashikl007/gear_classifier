# Convolutional Neural Network

# Importing the libraries
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print("TF version = ", tf.__version__)
import numpy as np
from keras.preprocessing import image
import cv2
import time
# Part 1 - Data Preprocessing

'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
'''


# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('D:/Projects/pythonProject/detector_for_industry/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 20,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('D:/Projects/pythonProject/detector_for_industry/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 20,
                                            class_mode = 'binary')

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))
# Part 3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# Training the CNN on the Training set and evaluating it on the Test set
start = time.time()
cnn.fit(x = training_set, validation_data = test_set, epochs = 20)
stop = time.time()
print("time required for training = ", stop-start)

#save model
i=str(1)
name = 'cnn'+i
path ='D:/Projects/pythonProject/detector_for_industr/'
model_name= path+name+'.h5'

model_name ="D:/Projects/pythonProject/detector_for_industry/cnn1.h5"
cnn.save(model_name)

#load model
model_name ="D:/Projects/pythonProject/detector_for_industry/cnn1.h5"
#detection using model
model = tf.keras.models.load_model(model_name)


#following script is used for tesing
def detect_obj(image):
    #test_image_conv = image.img_to_array(test_image)
    test_image = cv2.resize(image, (64, 64))
    test_image_conv = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image_conv)
    res = result.tolist()
    # training_set.class_indices
    prediction = 'no prediction'
    if res[0][0] == 1:
        prediction = 'blank'
    if res[0][1] == 1:
        prediction = 'herringbone'
    if res[0][2] == 1:
        prediction = 'spur'
    print(prediction)


# Part 4 - Making a single prediction
image_path = 'D:/Projects/pythonProject/detector_for_industry/herringbone_gear_360.jpg'

img = cv2.imread(image_path)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

test_image = cv2.resize(img, (64,64))
cv2.imshow('test_image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#test_image = image.load_img(img, target_size=(64, 64))
start_time = time.time()
detect_obj(test_image)
stop_time = time.time()
print("time required for detection = ", stop_time-start_time)


test_image_conv = np.expand_dims(test_image, axis=0)
result = model.predict(test_image_conv)
res = result.tolist()


cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    cv2.imshow("image", image)
    key = cv2.waitKey(20)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    detect_obj(image)

cv2.destroyAllWindows()
cap.release()




