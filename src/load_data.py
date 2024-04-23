import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_images(data_dir, image_size=(224, 224)):
    """
    Load images from the specified directory, resize them, and normalize the pixel values.
    :param data_dir: str, path to the dataset directory.
    :param image_size: tuple, the target size of the images.
    :return: tuple, numpy arrays of images and their labels.
    """
    images = []
    labels = []
    categories = os.listdir(data_dir)  # Assuming each category has its own folder

    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(int(category))  # Assuming folder names are labels

    return np.array(images), np.array(labels)

def prepare_dataset(data_dir, test_size=0.2, image_size=(224, 224)):
    """
    Load dataset, preprocess and split into training and testing sets.
    :param data_dir: str, the directory containing the dataset.
    :param test_size: float, the proportion of the dataset to include in the test split.
    :param image_size: tuple, the target size of the images.
    :return: tuple, training and testing data (X_train, X_test, y_train, y_test).
    """
    X, y = load_images(data_dir, image_size)
    y = tf.keras.utils.to_categorical(y)  # One-hot encode labels
    return train_test_split(X, y, test_size=test_size, random_state=42)
