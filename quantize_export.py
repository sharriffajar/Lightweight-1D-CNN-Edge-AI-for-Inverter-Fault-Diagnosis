"""
quantize_export.py
===================
Quantize model Keras terlatih (dari train_model.py) ke INT8 TFLite,
siap dipakai di ESP32-S3 lewat TFLite Micro.

Input:
  dataset_output/trained_model.keras
  dataset_output/X_train.npy   (dipakai sebagai representative dataset)

Output:
  dataset_output/model_int8.tflite
"""

import os
import numpy as np
import tensorflow as tf

DATA_DIR = 'dataset_output'
N_REP_SAMPLES = 200  # jumlah sampel representative dataset untuk kalibrasi


def representative_dataset_gen(X_train):
    def rep_dataset():
        for i in range(min(N_REP_SAMPLES, len(X_train))):
            yield [X_train[i:i + 1]]
    return rep_dataset


def quantize(model_path, X_train):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model


def save_model(tflite_model, out_path):
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Model saved: {out_path} ({size_kb:.1f} KB)")
    return size_kb


def main():
    model_path = os.path.join(DATA_DIR, 'trained_model.keras')
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))

    print("Quantizing for ESP32...")
    tflite_model = quantize(model_path, X_train)

    out_path = os.path.join(DATA_DIR, 'model_int8.tflite')
    save_model(tflite_model, out_path)

    print("\nFiles in dataset_output/:")
    for f in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, f)
        print(f"  - {f}: {os.path.getsize(fp) / 1024:.1f} KB")

    print("\nNext step: convert model_int8.tflite to a C array for firmware, e.g.:")
    print("  xxd -i dataset_output/model_int8.tflite > model_data.h")


if __name__ == '__main__':
    main()
