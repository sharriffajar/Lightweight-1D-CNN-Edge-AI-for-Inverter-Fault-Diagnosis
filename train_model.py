"""
train_model.py
===============
Training & evaluasi CNN 1D untuk klasifikasi fault (6 kelas) dari
window sinyal arus. Membaca dataset yang dihasilkan generate_dataset.py.

Bisa dijalankan langsung sebagai script, atau isi tiap fungsinya
ditempel ke sel-sel notebook (train_model.ipynb) kalau lebih suka
eksplorasi interaktif.

Input:
  dataset_output/X_data.npy
  dataset_output/y_data.npy

Output:
  dataset_output/trained_model.keras   <- dipakai oleh quantize_export.py
  confusion_matrix.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

WINDOW_SIZE = 128
NUM_CLASSES = 6
CLASS_NAMES = ['Healthy', 'S1_Open', 'S2_Open', 'S3_Open', 'S4_Open', 'Multi_Fault']
DATA_DIR = 'dataset_output'


def load_dataset():
    X = np.load(os.path.join(DATA_DIR, 'X_data.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_data.npy'))
    print(f"Loaded dataset - X: {X.shape}, y: {y.shape}")
    return X, y


def split_dataset(X, y):
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_tmp, axis=1)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model, X_train, y_train, X_val, y_val):
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    return history


def evaluate(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print("NOTE: this number is measured on the synthetic/toy dataset - see README disclaimer.")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - synthetic dataset (Acc: {test_acc * 100:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return test_acc


def main():
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    model = build_model()
    model.summary()

    train(model, X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test)

    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(DATA_DIR, 'trained_model.keras')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Keep a reference to X_train on disk for the representative dataset
    # used later in quantize_export.py
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)


if __name__ == '__main__':
    main()
