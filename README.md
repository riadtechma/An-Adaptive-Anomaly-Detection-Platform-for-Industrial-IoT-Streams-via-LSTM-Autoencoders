# An Adaptive Anomaly Detection Platform for Industrial IoT Streams

This repository provides the official implementation for the paper: **"An Adaptive Anomaly Detection Platform for Industrial IoT Streams via LSTM Autoencoders and Human-in-the-Loop Feedback: Supply Chain Application."**

Our platform bridges the gap between high-fidelity Deep Learning and resource-constrained industrial edge hardware by utilizing a lightweight LSTM Autoencoder (LSTM-AE) coupled with a novel **Targeted Local Overfitting (TLO)** mechanism.

## Core Components

* **`ai_model.py`**: Implementation of the core LSTM-AE architecture, featuring a 4-dimensional latent bottleneck and the `force_learn` algorithmic logic.
* **`dashboard.py`**: A real-time monitoring interface for visualizing sensor streams and reconstruction errors, including the interactive Human-in-the-Loop (HITL) trigger system.

## Dataset
The experimental dataset, consisting of 1.5 million sensor events with deterministically injected industrial fault patterns (e.g., transient spikes, persistent drift) overlaid on real-world healthy operational baselines, is available on Zenodo:

**The DataSet of the paper**: [https://zenodo.org/records/18382029](https://zenodo.org/records/18382029)

