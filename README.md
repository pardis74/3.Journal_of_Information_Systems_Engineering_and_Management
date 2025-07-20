# Ferroresonance Detection System (Conceptual)

This repository contains a conceptual implementation of a ferroresonance detection system using Discrete Wavelet Transform (DWT) feature extraction, a Long Short-Term Memory (LSTM) neural network for classification, and a Genetic Algorithm (GA) to optimize the LSTM hyperparameters.

---

## Overview

Ferroresonance detection in power systems is a challenging problem. This project simulates the detection process by:

- Generating dummy time-series data resembling DWT features from voltage waveforms.
- Using an LSTM model to classify signals as ferroresonance or normal operation.
- Applying a Genetic Algorithm to optimize the LSTM parameters, such as the number of hidden units and training epochs, balancing accuracy and training time.

The approach is inspired by concepts and configurations discussed in related research papers, including wavelet decomposition levels, LSTM structure, and GA settings.

---

## Key Components

### DataLoader

- Simulates loading and preparing time-series data.
- Generates dummy data shaped as `(samples, timesteps, features)`.
- Labels represent binary classes: ferroresonance (1) or normal (0).

### WaveletTransform

- Demonstrates Discrete Wavelet Transform (DWT) for signal decomposition.
- Uses `pywt` library to obtain approximation and detail coefficients.
- Intended for feature extraction from raw voltage signals (conceptual only).

### LSTMModel

- Builds and trains an LSTM neural network classifier using TensorFlow/Keras.
- Configurable number of hidden units and learning rate.
- Uses dropout for regularization and supports binary or multi-class classification.
- Evaluates the model with accuracy, precision, recall, and F1-score.

### GeneticAlgorithm

- Implements a Genetic Algorithm to optimize LSTM hyperparameters:
  - Number of hidden units (10 to 50)
  - Number of training epochs (5 to 15)
  - Weight coefficients for balancing accuracy and training time in fitness calculation.
- Uses roulette wheel style selection, single-point crossover, and mutation.
- Runs for a fixed number of generations and maintains the best solution found.

---

## Dependencies

- Python 3.7+
- numpy
- pywt
- scikit-learn
- tensorflow (tested with tensorflow 2.x)

Install dependencies via pip:

```bash
pip install numpy pywt scikit-learn tensorflow
