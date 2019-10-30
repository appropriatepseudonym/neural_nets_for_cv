from __future__ import print_function
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
#from keras.datasets import mnist
#import numpy as np
from keras.utils import to_categorical

#variables that i want to change on occasion
batch_size = 8
epochs = 10
lr=0.0001
decay=1e-9

'''import dataset'''
print ("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")

'''split data'''
(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, test_size=.25, random_state=42)

trainX = trainX.reshape(trainX.shape[0], 784)
testX = testX.reshape(testX.shape[0], 784)

'''set data as float and rescale pixel intensisty to lie between 0 and 1'''
trainX = trainX.astype('float32')/255
testX = testX.astype('float32')/255

'''change Y to one-hot'''
trainY_one_hot = to_categorical(trainY)
testY_one_hot = to_categorical(testY)

'''define the model'''
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation = "relu"))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation='softmax'))

print("[INFO] lr = {}, decay = {}" .format(lr,decay))

'''schocastic gradient decent activation and compile mode'''
sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

'''create history of training and create predictions'''
history = model.fit(trainX, trainY_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.125)
preds = model.evaluate(testX, testY_one_hot, batch_size=batch_size)

'''plot history metrics'''
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
