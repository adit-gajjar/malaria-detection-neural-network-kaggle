from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

#image processing library
import PIL as pillow

import keras 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
path = '/Users/vikasgajjar/Documents/machine_learning/malaria_detection/cell_images'
infected_inputs = [f for f in os.listdir('/Users/vikasgajjar/Documents/machine_learning/malaria_detection/cell_images/Parasitized') if not f.startswith('.')]
uninfected_inputs = [f for f in os.listdir('/Users/vikasgajjar/Documents/machine_learning/malaria_detection/cell_images/Uninfected') if not f.startswith('.')]

training_data = []
training_labels = [] # 1 = infected, 0 = uninfected

test_data = []
test_labels = [] # 1 = infected, 0 = uninfected

READ_DATA = False

num_of_training_examples = int(len(infected_inputs) * 0.7)
if (READ_DATA):
    counter = 0
    for infected in infected_inputs:
        try:
            curr_image = pillow.Image.open(path + '/Parasitized/' + infected)
                # shape of image_array is (64, 64, 3) 64x64 corresponds to resized image and 3 is for RGB values
            image_array = np.array(curr_image.resize((64, 64)))
            if (counter < num_of_training_examples):
                training_data.append(image_array)
                training_labels.append(1)
            else: # adding to our test data
                test_data.append(image_array)
                test_labels.append(1)
            counter += 1
        except AttributeError:
            print(AttributeError)
        
    counter = 0
    for uninfected in uninfected_inputs:
        try:
            curr_image = pillow.Image.open(path + '/Uninfected/' + uninfected)
            # shape of image_array is (64, 64, 3) 71x71 corresponds to resized image and 3 is for RGB values
            image_array = np.array(curr_image.resize((64, 64)))
            print(image_array.shape)
            if (counter < num_of_training_examples):
                training_data.append(image_array)
                training_labels.append(0)
            else: # adding to our test data
                test_data.append(image_array)
                test_labels.append(0)
            counter += 1
        except AttributeError:
            print(AttributeError)

    training_data = np.array(training_data)
    test_data = np.array(test_data)
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)
    print(training_data.shape)
    print(test_data.shape)
    print(training_labels.shape)
    print(test_labels.shape)

    np.save("training_data", training_data)
    np.save("test_data", test_data)
    np.save("training_labels", training_labels)
    np.save("test_labels", test_labels)


training_data = np.load('training_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)
training_labels = np.load('training_labels.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy', allow_pickle=True)
print(training_labels.shape)
training_labels = keras.utils.to_categorical(training_labels,2)
test_labels =keras.utils.to_categorical(test_labels ,2)
print(training_labels.shape)

base_model = keras.applications.resnet.ResNet50(include_top=False, 
                      weights='imagenet', 
                      input_shape=(64, 64, 3))
base_model.trainable = True #Tells that all the weights are to be trained
model = keras.models.Sequential()           #Creating the sequntial model
model.add(base_model)                #Adding the imagenet model
model.add(Flatten())                                #
model.add(Dense(512,activation="relu"))             #
model.add(Dropout(0.2))                             #
model.add(Dense(2,activation="softmax"))  #
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
his = model.fit(training_data,training_labels,batch_size=32, epochs= 1, validation_split=0.2)



model.save('cells.h5')