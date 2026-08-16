"""
generate_dataset.py
====================
Generator dataset sintetis (TOY) untuk validasi pipeline deteksi
open-circuit fault (OCF) pada inverter single-phase.

STATUS: Dataset di sini adalah data sintetis buatan tangan (scaling +
phase-shift per kelas), BUKAN hasil simulasi fisik (MATLAB/Simulink)
atau akuisisi hardware. Tujuannya murni untuk menguji apakah pipeline
generate -> train -> quantize -> export berjalan end-to-end.
Jangan jadikan akurasi dari dataset ini sebagai klaim performa deteksi
fault dunia nyata. Lihat README bagian "Status Proyek".

Output:
  dataset_output/X_data.npy
  dataset_output/y_data.npy
  dataset_validation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ========== CONFIGURATION ==========
WINDOW_SIZE = 128
NUM_CLASSES = 6
SAMPLES_PER_CLASS = 800
CLASS_NAMES = ['Healthy', 'S1_Open', 'S2_Open', 'S3_Open', 'S4_Open', 'Multi_Fault']

OUTPUT_DIR = 'dataset_output'


def generate_waveform(fault_type=0, seed=None):
    """Generate a single synthetic waveform window for a given fault class."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 2 * np.pi, WINDOW_SIZE)

    if fault_type == 0:
        signal = np.sin(t) + 0.05 * np.sin(3 * t)
    elif fault_type == 1:
        signal = np.sin(t).copy()
        signal[signal > 0] *= 0.18
    elif fault_type == 2:
        signal = np.sin(t).copy()
        signal[signal < 0] *= 0.18
    elif fault_type == 3:
        signal = np.sin(t + np.pi / 3).copy()
        signal[signal > 0] *= 0.22
    elif fault_type == 4:
        signal = np.sin(t + np.pi / 3).copy()
        signal[signal < 0] *= 0.22
    elif fault_type == 5:
        signal = (0.5 * np.sin(t) + 0.30 * np.sin(3 * t) +
                  0.20 * np.sin(5 * t) + 0.15 * np.random.randn(WINDOW_SIZE))
    else:
        raise ValueError("fault_type must be 0-5")

    signal += 0.02 * np.random.randn(WINDOW_SIZE)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    signal = (signal * 2.0) - 1.0

    return signal.astype(np.float32)


def build_dataset():
    print(f"Generating dataset - {NUM_CLASSES} classes, {SAMPLES_PER_CLASS} samples each")

    X_list, y_list = [], []
    for cls in range(NUM_CLASSES):
        for i in range(SAMPLES_PER_CLASS):
            w = generate_waveform(fault_type=cls, seed=i * 137 + cls * 53)
            X_list.append(w)
            y_list.append(cls)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}) done ({SAMPLES_PER_CLASS} samples)")

    X = np.array(X_list).reshape(-1, WINDOW_SIZE, 1)
    y = np.array(y_list)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_data.npy'), y)

    print(f"Dataset created - Shape: {X.shape}")
    return X, y


def plot_validation(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, ax in enumerate(axes.flat):
        sample = X[y == idx][0].flatten()
        ax.plot(sample, linewidth=2, color=f'C{idx}')
        ax.set_title(f'Class {idx}: {CLASS_NAMES[idx]}')
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.3, 1.3)

    plt.suptitle('Dataset Validation (SYNTHETIC/TOY) - not representative of real OCF signatures')
    plt.tight_layout()
    plt.savefig('dataset_validation.png')
    plt.show()
    print("Saved dataset_validation.png")


if __name__ == '__main__':
    X, y = build_dataset()
    plot_validation(X, y)
