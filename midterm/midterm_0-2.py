from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

'''unused'''
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import glob, os, cv2

'''model definition source'''
import models

'''variables i might want to change on a regular basis'''
img_dir="imgs"
batch_size = 1
epochs = 10
lr=0.01
decay=1e-9
ncols = 227
nrows = 227
ndims = 3
input_shape = (nrows, ncols, ndims)
#nclasses = len(np.unique())

'''get img locations'''
data_dir = os.path.join(os.getcwd(), img_dir)
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")


'''normalize images'''
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(ncols,nrows),
                                                    shuffle = True,
                                                    class_mode = 'categorical')

generator_test = datagen_test.flow_from_directory(  directory=test_dir,
                                                    batch_size=batch_size,
                                                    target_size=(ncols, nrows),
                                                    class_mode = 'categorical',
                                                    shuffle = False)

nclasses = len(generator_train.class_indices)
steps_test = generator_test.n // batch_size
print(steps_test)


steps_per_epoch = generator_train.n // batch_size
print(steps_per_epoch, "Steps Per Epoch")

#model = AlexNet(input_shape, nclasses)
model = models.custom(input_shape, nclasses)
model.summary()

Adam = keras.optimizers.adam(lr=lr, amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer= Adam,\
 metrics=['accuracy'])

history= model.fit_generator(generator_train,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_test,
                           validation_steps = steps_test)

'''show me the plots'''
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