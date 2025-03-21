# Aircraft Detection using YOLO11

This project demonstrates a step-by-step guide to building an object detection program for aircraft detection using YOLO11. The notebook provides instructions for setting up the environment, preparing the dataset, configuring the model, and training it locally. This project is intended for academic and learning purposes.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Requirements](#setup-and-requirements)
3. [Steps](#steps)
   - [Step 1: Preliminaries, Environment, and Required Packages](#step-1-preliminaries-environment-and-required-packages)
   - [Step 2: Dataset and Organization](#step-2-dataset-and-organization)
   - [Step 3: Setting up Labels](#step-3-setting-up-labels)
   - [Step 4: Training Configuration](#step-4-training-configuration)
   - [Step 5: Training the Model](#step-5-training-the-model)
4. [References](#references)

---

## Introduction

This project uses YOLO11, a state-of-the-art object detection model, to detect aircraft in images. The dataset includes annotations in GeoJSON format, and the notebook walks through the process of preparing the data, configuring the model, and training it locally using NVIDIA GPUs.

## Setup and Requirements

### Required Libraries
The following Python main libraries are required for this project:
- `matplotlib`
- `numpy`
- `Pillow`
- `ultralytics`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended for training)
- Sufficient storage for the dataset and model weights

---

## Steps

### Step 1: Preliminaries, Environment, and Required Packages
Set up the environment and install the required Python libraries. Ensure that your system has the necessary dependencies for YOLO11 and GPU acceleration.

### Step 2: Dataset and Organization
Organize the dataset, which includes images and annotations in GeoJSON format. The dataset should be structured to allow easy access for training and evaluation.

### Step 3: Setting up Labels
Understand the dataset's annotation format (GeoJSON polygons) and process the labels to prepare them for training.

### Step 4: Training Configuration
Configure the YOLO11 model for training. This includes setting up the model architecture, defining hyperparameters, and specifying the dataset paths.

### Step 5: Training the Model
Train the YOLO11 model using the prepared dataset and configuration. Monitor the training process and evaluate the model's performance.

---

## References
1. EJ Technology Consultants (2024). [How to Train YOLO 11 Object Detection Models Locally with NVIDIA](https://www.ejtech.io/learn/train-yolo-models#:~:text=This%20guide%20provides%20step-by-step%20instructions%20for%20training%20a,on%20a%20local%20PC%20using%20an%20NVIDIA%20GPU.).
2. Ultralytics. (2024). [YOLO Documentation](https://docs.ultralytics.com/).
3. Database [Airbus Aircraft Detection](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset) on Kaggle.

---