from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical

"""
def convolution_layer(input_layer, filters, kernel_size=[3, 3], activation='relu'):
	layer = tf.layers.conv2d(
		inputs=input_layer,
		filters=filters,
		kernel_size=kernel_size,
		activation=activation,
	)
	add_variable_summary(layer, 'convolution')
	return layer
"""

def createModel():
    
    #784-256-128-10
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding ='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))	
	
    model.add(Conv2D(64, (3,3), padding ='same', activation='relu')) 
    model.add(Conv2D(64, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
 
    '''
    model.add(Conv2D(64, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    '''
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))	

    return model

#variables that i want to change on occasion
batch_size = 256
epochs = 5
lr=0.01
decay=1e-3

#import dataset
print ("[INFO] loading MNIST (full) dataset...")

#for 1d?
'''
#pull data
dataset = datasets.fetch_mldata("MNIST Original")

#split data
(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, test_size=.25, random_state=42)
'''

#for 2d - pull data and split
(trainX, trainY), (testX, testY) = mnist.load_data()


nrows, ncols = trainX.shape[1:]
ndims=1
trainX = trainX.reshape(trainX.shape[0], nrows, ncols, 1)
testX = testX.reshape(testX.shape[0], nrows, ncols, 1)

#print("Training data shape : ", dataset.data.shape, dataset.target.shape)
input_shape = (nrows, ncols, ndims)
#set data as float and rescale pixel intensisty to lie between 0 and 1
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

trainY_one_hot = to_categorical(trainY)
testY_one_hot = to_categorical(testY)

model1=createModel()
print("[INFO] lr = {}, decay = {}" .format(lr,decay))
sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

history = model1.fit(trainX, trainY_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY_one_hot))
model1.evaluate(testX, testY_one_hot)

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
