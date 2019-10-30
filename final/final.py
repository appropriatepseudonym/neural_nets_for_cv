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
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import glob, os, cv2

#model definition source
import models

#variables i might want to change on a regular basis
img_dir="classification"
batch_size =2
epochs = 200
lr=0.0001
decay=1e-9

#these are for image resize
ncols = 227
nrows = 227
ndims = 3
input_shape = (nrows, ncols, ndims)

img_list = list(glob.glob('**/*.jpg', recursive=True))
#initialize dataset and label lists
data_set = [] 
label = []
print("adding images...")
for i in img_list:
    image = cv2.imread(i)
    data_set.append(cv2.resize(image,(ncols,nrows)))
    label.append(os.path.split(os.path.split(i)[-2])[-1])
print("splitting data into train, validation, and test")
(trainX, valX, trainY, valY) = train_test_split(data_set, label, test_size=.20, random_state=0)

encoder = LabelBinarizer()
trainY_one_hot = encoder.fit_transform(trainY)
valY_one_hot = encoder.fit_transform(valY)

print('training data size', len(trainX))
print('validation data size', len(valX))

'''get img locations
data_dir = os.path.join(os.getcwd(), img_dir)
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
'''

'''normalize images'''
datagen_train = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  #validation_split=0.2
                                  )
datagen_val = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow(x=np.asarray(trainX),
                                    y=trainY_one_hot,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    #class_mode = 'categorical'
                                    #subset='training'
                                    )
generator_val = datagen_val.flow(x=np.asarray(valX),
                                    y=valY_one_hot,
                                    batch_size = batch_size,
                                    shuffle = False,
                                    #class_mode = 'categorical'
                                    )
'''
generator_train = datagen_train.flow_from_directory(directory=data_dir,
                                                    batch_size=batch_size,
                                                    target_size=(ncols,nrows),
                                                    color_mode='grayscale',
                                                    shuffle = True,
                                                    class_mode = 'categorical',
                                                    subset='validation')

generator_val = datagen_train.flow_from_directory(directory=data_dir,
                                                 batch_size=batch_size,
                                                 target_size=(ncols, nrows),
                                                 color_mode='grayscale',                                                    class_mode = 'categorical',
                                                 shuffle = False,
                                                 subset='validation')
'''

nclasses = len(np.unique(trainY))
steps_val = generator_val.n // batch_size
print(steps_val)


steps_per_epoch = generator_train.n // batch_size
print(steps_per_epoch, "Steps Per Epoch")

#model = AlexNet(input_shape, nclasses)
model = models.AlexNet(input_shape, nclasses)
model.summary()

Adam = keras.optimizers.adam(lr=lr, amsgrad = True)
model.compile(loss='categorical_crossentropy', optimizer= Adam,\
 metrics=['accuracy'])

history= model.fit_generator(generator_train,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data = generator_val,
                           validation_steps = steps_val)

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


preds = model.evaluate_generator(generator_test, steps=steps_per_epoch)
print("test image loss={:.7f}, acc={:.7f}" .format(preds[0], preds[1]))