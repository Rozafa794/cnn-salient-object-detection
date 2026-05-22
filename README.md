# CNN-Based Salient Object Detection

This project develops and evaluates a CNN-based model for Salient Object Detection using a subset of the ECSSD dataset.

## Thesis Title

Zhvillimi dhe vlerësimi i një modeli CNN për detektimin e objekteve të spikatura në imazhe

## Project Description

The goal of this project is to develop a convolutional neural network model that detects salient objects in images. The model takes an input image and predicts a saliency mask that highlights the most important object in the image.

## Dataset

The dataset used in this project is ECSSD (Extended Complex Scene Saliency Dataset).

Dataset links:

- Images: https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip
- Ground Truth Masks: https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip

In this project, a subset of 100 image-mask pairs from the ECSSD dataset was used for initial training and evaluation.

## Project Structure

```text
sod_project/
│
├── data_loader.py
├── evaluate.py
├── sod_model.py
├── train.py
├── utils.py
├── results/
├── splits/
├── checkpoints/
└── dataset/