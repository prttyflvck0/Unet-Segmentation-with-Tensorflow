import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from unet import build_unet


""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.normpath(os.path.join(path, '*.jpg'))))
    train_y = sorted(glob(os.path.normpath(os.path.join(path, '*.png'))))

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_colormap():
    num_labels = 13

    # BGR format
    colormap = [
        [0, 0, 0],
        [170, 0, 0],
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 85],
        [0, 0, 170],
        [85, 255, 170],
        [0, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [0, 255, 255]
    ]

    classes = [
        "unknown",
        "background",
        "facade",
        "window",
        "door",
        "cornice",
        "sill",
        "balcony",
        "blind",
        "pillar",
        "deco",
        "molding",
        "shop"
    ]

    return classes, colormap


def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))

    """ Mask processing """
    output = []
    for color in COLORMAP:
        cmap = np.all(np.equal(x, color), axis=-1)
        output.append(cmap)

    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)

    return output


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory to save files """
    create_dir("files")

    """ Hyper parameters """
    IMG_H = 320
    IMG_W = 416
    NUM_CLASSES = 13  # classes + unknown class
    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 4
    lr = 1e-4  # 0.0001
    num_epochs = 100

    dataset_path = 'CMP_facade_DB_base/base'  # only eng letters
    model_path = os.path.normpath(os.path.join("files", 'model.h5'))
    csv_path = os.path.normpath(os.path.join("files", 'data.csv'))

    print(f'Tensorflow version {tf.__version__}')
    print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF"}')

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(
        f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Process colormap """
    CLASSES, COLORMAP = get_colormap()

    """ Dataset pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape, NUM_CLASSES)
    model.compile(
        loss='categorical_crossentropy',  # binary_crosscentropy for binary segmentation
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    checkpoint_filepath = os.path.normpath(r'tmp')
    callbacks = [
        ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=num_epochs,
              callbacks=callbacks
              )

    print('finished')
