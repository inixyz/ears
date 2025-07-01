# 🎧 EARS: Environmental Acoustic Response Simulator

**EARS** is a GPU-accelerated 3D acoustics simulation framework that models physically accurate sound propagation using Finite-Difference Time-Domain (FDTD) schemes.

Developed as part of a diploma thesis at the National University of Science and Technology POLITEHNICA Bucharest, this software is designed for researchers, engineers, and developers interested in wave-based audio simulation, architectural acoustics, immersive media, and dataset generation for machine learning.

---

## 🧭 Overview

EARS simulates sound propagation in complex three-dimensional environments, supporting:

- ✅ Reflection, refraction, diffraction, absorption
- ✅ Multi-material modeling with varying impedance
- ✅ Realistic open/closed boundary conditions
- ✅ GPU-accelerated performance via **CUDA**
- ✅ Python scripting interface for full control
- ✅ Real-time acoustic field **visualization**

---

## 🎯 Features

- 🎧 **Wave-based simulation** using FDTD schemes
- 🚀 **High-performance CUDA acceleration**
- 🧱 **Custom materials and multi-surface models**
- 🔌 **Python API** for scripting, testing, and dataset creation
- 📈 **Interactive visualization** of acoustic wavefields
- 🎵 **Room Impulse Response (RIR)** generation and export

---

## 📸 Demo

![Simulation Visualization](assets/demo.gif)

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyBind11 installed
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed and in PATH
