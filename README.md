 Face Age Prediction with ResNet50

A computer vision model that predicts a person’s age from a facial image using a fine-tuned ResNet50 architecture. Built as part of a supervised learning project to support real-world applications like age verification at retail checkout.

## Project Overview

This project explores how deep learning can be used to estimate a person's age from facial images. The goal was to train a model that achieves a mean absolute error (MAE) below 8 years — a threshold suitable for assisting with alcohol sale compliance in retail environments.

The model was trained using a dataset of 7.6k labeled face images, and validated using a separate subset. We used Keras's `ImageDataGenerator` for efficient image loading and augmentation, and a transfer learning approach using ResNet50 as the base model.

## Key Results

- ✅ Final validation MAE: **6.64**
- ✅ Requirement met: MAE ≤ 8
- 🧠 Model generalizes well with minimal overfitting
- 🧪 Tested with different learning rates and model configurations

## Model Architecture

- Pretrained base: **ResNet50** (ImageNet weights, frozen initially)
- Global average pooling layer
- Dense layer with 128 ReLU units
- Final `Dense(1)` output for regression
- Optimizer: Adam with `learning_rate=0.0005`
- Loss: Mean Squared Error
- Metric: Mean Absolute Error

## Dataset

- ~7,600 facial images with real-valued age labels
- Labels provided in a CSV alongside image filenames
- Data split: 75% training, 25% validation
- Images were resized to **224×224** for ResNet compatibility

> 📦 The dataset was provided through a bootcamp platform and is based on [ChaLearn LAP's apparent age dataset](https://chalearnlap.cvc.uab.cat/dataset/26/data/45/description/). It is not publicly redistributed in this repo due to size and license constraints.

## How to Run

1. Clone the repo and unzip the dataset into a folder called `faces/`, with:
   ```
   ./faces/
   ├── labels.csv
   └── final_files/
       ├── 00001.jpg
       ├── ...
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow pandas matplotlib
   ```

3. Run the training script or notebook:
   ```bash
   python run_model_on_gpu.py
   ```

Or open the notebook:

```bash
age_prediction_with_resnet50.ipynb
```

## Use Case

This model could be used as part of a decision-support system at retail checkout counters to assist with age verification. While not a replacement for ID checks, it could reduce manual errors and flag potentially underage customers in real time.

## Future Work

- Add dropout layers to reduce overfitting
- Explore lightweight architectures (e.g., MobileNetV2) for edge deployment
- Improve generalization through data augmentation and fine-tuning
