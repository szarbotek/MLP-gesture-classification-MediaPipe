"""
Module responsible for training the machine learning model on prepared data.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from src import Config

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.logs import logs


def activate_train(
        points,
        labels,
        save_project_relative_path=logs.path("data/test/model"),
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64]
]:
    """
    Prepares model configuration, processes input labels, and executes training.

    :param points: Array containing input feature points.
    :param labels: Array containing raw class labels.
    :param save_project_relative_path: Save path.
    :param save_names: Base name used for saving the output trained model.
    :return: Split datasets
        (train_labels, train_points, test_labels, test_points,
        valid_labels, valid_points).
    """

    # ---------------------------------------------------------
    # Save path
    # ---------------------------------------------------------
    save_project_relative_path = save_project_relative_path / "model"
    save_project_relative_path.mkdir(parents=True, exist_ok=True)

    sw = logs.stopwatch("Model Training")
    sw.run()

    logs.print("p-s", "TensorFlow model training process started")

    # ---------------------------------------------------------
    # Label encoding
    # ---------------------------------------------------------
    encoder = LabelEncoder()

    # Map string labels to numeric values
    label_transform = encoder.fit_transform(labels)

    # Generate one-hot encoded vectors
    onehot = tf.keras.utils.to_categorical(label_transform)

    # Save fitted encoder classes
    logs.storage.save.numpy( "classes", save_project_relative_path,encoder.classes_ )

    logs.print(   "i",   f"Base data shape: labels={labels.shape}, points={points.shape}")

    # ---------------------------------------------------------
    # Data separation
    # ---------------------------------------------------------
    def data_separate(points, labels):

        # Test: 15%
        rest_points, test_points, rest_labels, test_labels = train_test_split(
            points,
            labels,
            test_size=0.15,
            stratify=labels,
            random_state=42
        )

        # Validation: 15%
        # 17.65% of remaining 85% ≈ 15% of total
        train_points, valid_points, train_labels, valid_labels = train_test_split(
            rest_points,
            rest_labels,
            test_size=0.1765,
            stratify=rest_labels,
            random_state=42
        )

        return (
            train_labels, train_points,
            test_labels, test_points,
            valid_labels, valid_points
        )

    # Split dataset into train, test, and validation sets
    (
        train_labels, train_points,
        test_labels,  test_points,
        valid_labels, valid_points
    ) = data_separate(points, onehot)

    logs.print(
        "i",f"Train data shape: labels={train_labels.shape}, points={train_points.shape}"
    )

    logs.print(
        "i",f"Test data shape: labels={test_labels.shape}, points={test_points.shape}"
    )

    logs.print(
        "i",f"Valid data shape: labels={valid_labels.shape}, points={valid_points.shape}"
    )

    # ---------------------------------------------------------
    # Model architecture
    # ---------------------------------------------------------
    model = Sequential([
        Dense(
            128,
            activation="relu",
            input_shape=(63,)
        ),
        BatchNormalization(),
        Dropout(0.4),

        Dense(
            256,
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(
            128,
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(0.3),

        Dense(
            len(encoder.classes_),
            activation="softmax"
        )
    ])
    # ---------------------------------------------------------
    # Model compilation
    # ---------------------------------------------------------
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # ---------------------------------------------------------
    # Training callbacks
    # ---------------------------------------------------------
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    # ---------------------------------------------------------
    # Model training
    # ---------------------------------------------------------
    run = model.fit(
        train_points,
        train_labels,
        validation_data=(
            valid_points,
            valid_labels
        ),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------------------------------------------------
    # Model saving
    # ---------------------------------------------------------
    model_save_path = ( save_project_relative_path / f"gesture_classifier.keras" )
    model.save( Config.PROJECT_ROOT / model_save_path )
    logs.print("f-s", f"Model saved: {model_save_path}" )

    # ---------------------------------------------------------
    # Save history
    # ---------------------------------------------------------
    logs.storage.save.json(
        "history", save_project_relative_path, run.history
    )
    #
    logs.storage.save.numpy(
        "train_points", save_project_relative_path, train_points
    )
    logs.storage.save.numpy(
        "train_labels", save_project_relative_path, train_labels
    )
    logs.storage.save.numpy(
        "valid_points", save_project_relative_path, valid_points
    )
    logs.storage.save.numpy(
        "valid_labels", save_project_relative_path, valid_labels
    )
    logs.storage.save.numpy(
        "test_points", save_project_relative_path, test_points
    )
    logs.storage.save.numpy(
        "test_labels", save_project_relative_path,test_labels
    )

    sw.stop()

    logs.print( "p-e","Model training process finished successfully" )

    return (
        train_labels,
        train_points,
        test_labels,
        test_points,
        valid_labels,
        valid_points
    )