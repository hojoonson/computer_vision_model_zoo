import logging
import sys
import tensorflow as tf

from util.logger import set_stream_logger
from vgg.train import train_vgg16

if __name__ == '__main__':
    set_stream_logger('main_logger', logging.INFO,
                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    model_name = sys.argv[1]
    mode = sys.argv[2]
    
    if model_name == 'vgg16':
        if mode == 'train':
            train_vgg16()
    else:
        pass
