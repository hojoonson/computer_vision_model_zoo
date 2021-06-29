import numpy as np
import datetime
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from sklearn.utils import class_weight

from vgg.build_model import build_vgg16_model


TRAIN_DATA_DIR = os.path.join('datasets', 'vgg16', 'train')
VALIDATION_DATA_DIR = os.path.join('datasets', 'vgg16', 'val')
MODEL_SAVE_DIR = os.path.join('models', 'vgg16')

INPUT_SHAPE = (224, 224, 3)
OUTPUT_SHAPE = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.001
LEARNING_DECAY = 1e-6
STEP_FOR_EPOCH = 100
EPOCH = 100


def load_data_generator(data_dir):
    # add constructor parameter for augmentation
    img_data_gen = ImageDataGenerator()
    data_generator = img_data_gen.flow_from_directory(
        data_dir, batch_size=BATCH_SIZE, class_mode='categorical', target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
    return data_generator


def save_model(model, save_dir):
    now_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(save_dir, '_'.join([now_datetime]))
    os.mkdir(model_path)
    model.save(model_path)


def train_vgg16():    
    # load dataset
    train_data_generator = load_data_generator(TRAIN_DATA_DIR)
    validation_data_generator = load_data_generator(VALIDATION_DATA_DIR)

    # build model
    model = build_vgg16_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model.compile(optimizers.RMSprop(lr=LEARNING_RATE, decay=LEARNING_DECAY),
                loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # calculate class weight
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data_generator.classes),
        y=train_data_generator.classes)
    class_weights = dict(enumerate(class_weights))

    # train model
    try:
        model.fit(
            train_data_generator,
            validation_data=validation_data_generator,
            steps_per_epoch=STEP_FOR_EPOCH,
            verbose=1,
            epochs=EPOCH,
            class_weight=class_weights)
    except KeyboardInterrupt:
        print('keyboard interrupt on model fitting')
    finally:
        save_model(model, MODEL_SAVE_DIR)
