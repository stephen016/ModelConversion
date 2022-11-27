import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

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

# imagenette2
def get_imagenette_data(batch_size):
    data_dir = "data/imagenette2/train"
    data_dir_test ="data/imagenette2/val"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode='categorical',
        subset="training",
        seed=1234,
        image_size=(224, 224),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode='categorical',
        subset="validation",
        seed=1234,
        image_size=(224, 224),
        batch_size=batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir_test,
        label_mode='categorical',
        seed=1234,
        image_size=(224, 224),
        batch_size=batch_size)
    return train_ds,val_ds,test_ds
