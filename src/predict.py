import cv2
import numpy as np
import tensorflow as tf

def load_and_prepare_image(image_path, target_size=(224, 224)):
    """
    Load an image file and prepare it for prediction.
    :param image_path: str, path to the image file.
    :param target_size: tuple, target size of the image (height, width).
    :return: numpy array, processed image.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # Resize the image to match the model's input size
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand the dimensions to match model input
    return img

def predict_coin(image_path, model_path='models/euro_coin_model.h5'):
    """
    Predict the denomination of a Euro coin from an image.
    :param image_path: str, path to the coin image.
    :param model_path: str, path to the trained model file.
    :return: int, predicted class label.
    """
    model = tf.keras.models.load_model(model_path)  # Load the trained model
    image = load_and_prepare_image(image_path)  # Prepare the image
    prediction = model.predict(image)  # Make a prediction
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    return predicted_class

if __name__ == '__main__':
    # Example usage
    image_path = 'path_to_your_test_image.jpg'
    predicted_class = predict_coin(image_path)
    print(f'Predicted class: {predicted_class}')
