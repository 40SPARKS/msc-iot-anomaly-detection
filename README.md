# MSc IoT Anomaly Detection Project

This repository contains the data-processing and machine-learning pipeline for my MSc project on temporal anomaly detection in IoT-based motor/conveyor systems.

## Project Aim

The project investigates anomaly detection using sensor data from embedded IoT nodes. The planned system uses ESP32-based sensing nodes, MQTT communication, Raspberry Pi data logging, and machine-learning models such as Isolation Forest and GRU.

## Current Progress

Conda environment created for data processing.
Public rotating machine dataset sample prepared locally.
CSV preview script completed.
Window-based vibration feature extraction completed.
Initial Isolation Forest baseline implemented.

## Folder Structure

```text
MSc_Project_Data/
├── datasets/
│   └── public_rotating_machine/
│       ├── raw_part1/
│       ├── selected_sample/
│       └── processed/
├── esp32_platformio/
└── ml_pipeline/
    ├── 01_preview_dataset.py
    ├── 02_extract_window_features.py
    ├── 03_isolation_forest.py
    ├── 04_gru_preparation.py
    ├── requirements.txt
    ├── models/
    ├── notebooks/
    └── outputs/
```
