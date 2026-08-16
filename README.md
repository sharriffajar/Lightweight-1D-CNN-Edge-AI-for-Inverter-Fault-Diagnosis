<p align="center">
  <a href="README.md">🇬🇧 English</a> | 
  <a href="README.id.md">🇮🇩 Bahasa Indonesia</a>
</p>

# ⚡ Lightweight CNN Fault Detector — ESP32-S3

**Open-Circuit Fault Detection for Single-Phase Inverters using a Lightweight 1D CNN + TinyML**

> Part of a undergraduate thesis and scientific paper *"Democratizing AIoT for Renewable Energy Reliability"* — Electrical Engineering Department, Universitas Tanjungpura, Pontianak, Indonesia.

[![Status](https://img.shields.io/badge/status-proof--of--concept-yellow)]()
[![Platform](https://img.shields.io/badge/platform-ESP32--S3-blue)]()
[![Model](https://img.shields.io/badge/model-TFLite%20Micro%20INT8-orange)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 🎯 Overview

This repo contains a machine learning pipeline to detect **open-circuit fault (OCF)** conditions in a single-phase full-bridge inverter, designed to run directly on an **ESP32-S3** microcontroller (edge inference, no cloud/GPU dependency).

Pipeline:

```
Generate/Acquire Signal → Preprocessing → 1D CNN Training → INT8 Quantization → TFLite Micro → Deploy to ESP32-S3
```

Target specs: model < 200 KB, RAM < 100 KB, inference latency < 500 ms — aimed at residential/small-business solar PV systems that can't afford commercial monitoring solutions (USD 300-1000+ per unit).

---

## ⚠️ Project Status — Please Read Before Judging Any Numbers in This Repo

This repo is at the **proof-of-concept / pipeline feasibility** stage, not a system validated against real inverter faults. Please note:

| Component | Status | Note |
|---|---|---|
| Pipeline (generate → train → quantize → export) | ✅ Validated end-to-end | Runs fully, produces a quantized model in ESP32-ready format |
| Current dataset | 🧪 **Synthetic/toy** | Signals are hand-crafted (per-class amplitude scaling + phase shift) to test the pipeline — **not** the output of a physical circuit simulation or hardware acquisition |
| Accuracy on the toy dataset | 🚫 Not representative | Because each class is geometrically distinct by construction, almost any model separates them easily — this number **does not reflect** real-world fault-detection performance and is intentionally not presented as a result claim |
| Physical dataset (Simulink / lab hardware) | 🔜 Planned next step | See [Roadmap](#️-roadmap) |

**In short:** what's proven so far is that *the architecture and pipeline work and are deployable*. What's *not yet* proven is *real-world fault detection capability* — that claim will only be made once a physical dataset is available.

---

## 🏗️ Architecture

- **Model:** Lightweight 1D CNN (Conv1D → MaxPool → Conv1D → MaxPool → Conv1D → GAP → Dense → Softmax)
- **Input:** current signal window, 128 samples (≈128 ms @ 1 kHz sampling)
- **Output:** 6 classes — `Healthy`, `S1_Open`, `S2_Open`, `S3_Open`, `S4_Open`, `Multi_Fault`
- **Optimization:** post-training INT8 quantization via representative dataset, target model size < 200 KB
- **Target deployment:** ESP32-S3 (dual-core Xtensa LX7 @ 240 MHz), ACS712 current sensor, remote monitoring via Thinger.io

---

## 📁 Repo Structure

```
.
├── generate_dataset.py      # Synthetic signal generator (toy dataset, see disclaimer)
├── train_model.py           # 1D CNN training + evaluation
├── quantize_export.py       # Convert to TFLite Micro INT8
├── dataset_output/
│   ├── X_data.npy
│   ├── y_data.npy
│   └── model_int8.tflite
├── figures/
│   ├── dataset_validation.png
│   └── confusion_matrix.png
└── README.md
```

---

## 📊 Visualizations (Toy Dataset)

**Per-class signal shape validation:**
![Dataset Validation](figures/dataset_validation.png)

**Confusion matrix from training on the synthetic dataset:**
![Confusion Matrix](figures/confusion_matrix.png)

> ⚠️ The 100% accuracy above is measured on the synthetic/toy dataset, not an indicator of real-world fault-detection performance. See [Project Status](#️-project-status--please-read-before-judging-any-numbers-in-this-repo).

---

## 🗺️ Roadmap

- [x] Generate → train → quantize → export pipeline (proof-of-concept)
- [ ] Physics-based H-bridge inverter simulation (MATLAB/Simulink) for a more representative dataset
- [ ] Lab data acquisition: 500VA full-bridge inverter + ACS712 + physical OCF injection on IGBTs
- [ ] Re-evaluate accuracy & F1-score on the physical/lab dataset
- [ ] Real-time deployment & testing on ESP32-S3
- [ ] Thinger.io integration for remote monitoring + offline failover

---

## 🤝 Feedback Welcome

This is a student research project and I'd genuinely appreciate input from anyone with power-electronics, embedded ML, or TinyML experience — especially on the physical fault modeling step (see Roadmap). Feel free to open an issue or start a discussion.

---

## 📄 Related

This repo accompanies the scientific paper *"Democratizing AIoT for Renewable Energy Reliability: A Lightweight CNN and ESP32-S3-Based Inverter Fault Detection System"* — Independence Day Special Scientific Writing Competition (PT Borneo Alumina Indonesia), August 2026.

## 👤 Author

**Sharrif Faqih Fajarudin**
Student ID D1021221062 — Electrical Engineering, Faculty of Engineering, Universitas Tanjungpura, Pontianak, Indonesia

## 📜 License

MIT — free to use/modify with attribution.
