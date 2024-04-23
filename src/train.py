import os
from model import create_cnn_model
from load_data import prepare_dataset

def train_model(data_dir, model_save_path='models/euro_coin_model.h5', epochs=10, batch_size=32):
    """
    Train the CNN model on the Euro coin dataset.
    :param data_dir: str, path to the dataset directory.
    :param model_save_path: str, path where the trained model will be saved.
    :param epochs: int, number of epochs to train the model.
    :param batch_size: int, size of the batches of data.
    """
    # Load and prepare the dataset
    X_train, X_test, y_train, y_test = prepare_dataset(data_dir)

    # Create the CNN model
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=6)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)

    # Save the model
    model.save(model_save_path)

    # Optionally, return the history to analyze training/validation progress
    return history

if __name__ == '__main__':
    # Example: training the model on your dataset
    data_dir = 'path_to_your_dataset'
    model_path = 'models/euro_coin_model.h5'
    history = train_model(data_dir, model_path)

    # After training, you might want to display or analyze the training progress
    # For example, print the history of accuracy and loss
    print(history.history['accuracy'])
    print(history.history['loss'])
