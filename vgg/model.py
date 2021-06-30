from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.core import Dropout, Flatten


def vgg_16(input_shape=(224, 224, 3), output_shape=1000):
    """[summary]
    build VGG-16 model structure (D version)
    1. all padding size: 1
    2. all filter size: 3 x 3
    3. all activation function: relu
    4. all max pool size: 2 x 2  (stride=2) 
    Args:
        input_shape (tuple, optional): input shape of model. Defaults to (224, 224, 3).
        output_shape (int, optional): output shape of model. Defaults to 1000.
    Returns:
        model (instance): VGG-16 model
    """
    model = Sequential()
    
    # layer 1 ~ 2 (filter: 64)
    model.add(Input(shape=input_shape))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # output size: 112 x 112 x 64
    
    # layer 3 ~ 4 (filter: 128)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # output size: 56 x 56 x 128
    
    # layer 5 ~ 7 (filter: 256)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # output size: 28 x 28 x 256
    
    # layer 8 ~ 10 (filter: 512)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # output size: 14 x 14 x 512
    
    # layer 11 ~ 13 (filter: 512)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    # output size: 7 x 7 x 512
    
    # layer 14 ~ 16 (Fully Connected)
    model.add(Flatten())
    # flatten: 7 x 7 x 512 = 25,088
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    # categorized by output shape
    
    return model