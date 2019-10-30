from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization

def LeNet(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		#if K.image_data_format() == "channels_first":
		#	inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
    
def custom(input_shape, nclasses):
    
    #784-256-128-10
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding ='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(4096, activation='relu'))	

    model.add(Conv2D(64, (3,3), padding ='same', activation='relu')) 
    model.add(Conv2D(128, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))

    model.add(Conv2D(128, (3,3), padding ='same', activation='relu')) 
    model.add(Conv2D(396, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    
    model.add(Conv2D(256, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))

    model.add(Conv2D(1024, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(1024, (3,3), padding ='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(4096, activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(nclasses, activation='softmax'))	

    return model

def VGG_16(input_shape, nclasses, weights_path=None):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding ='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding ='same', activation='relu'))
    #model.add(Conv2D(256, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding ='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def AlexNet(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding = 'same', input_shape = input_shape))
    #model.add(BatchNormalization((64,229,229)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))
    
    model.add(Conv2D(128, (3,3), padding = 'same'))
    #model.add(BatchNormalization((128,117,117)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))
    
    model.add(Conv2D(192, (3,3), padding = 'same'))
    #model.add(BatchNormalization((128,115,115)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))
    
    model.add(Conv2D(256, (3,3), padding = 'same'))
    #model.add(BatchNormalization((128,110,110)))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3)))
    
    model.add(Flatten())
    model.add(Dense(12*12*256))
    #model.add(BatchNormalization(4096))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dense(4096))
    #model.add(BatchNormalization(4096))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dense(classes))
    #model.add(BatchNormalization(1000))
    model.add(BatchNormalization())    
    model.add(Activation('softmax'))
    return model
 