import tensorflow as tf
import numpy as np


# Cifar
def get_cifar_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0 
    x_train=x_train.astype("float32")
    x_test = x_test.astype("float32")

    train_num = int(x_train.shape[0]*0.8)
    x_val = x_train[train_num:]
    y_val = y_train[train_num:]
    x_train = x_train[:train_num]
    y_train = y_train[:train_num]
    return x_train,y_train,x_val,y_val,x_test,y_test
