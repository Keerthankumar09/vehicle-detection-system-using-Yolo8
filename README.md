# vehicle-detection-system-using-Yolo8
ai virtual internship AIMERS 

# 🚀 Train YOLOv8 on a Custom Dataset

This repository contains a Colab-based notebook for training YOLOv8 on a custom object detection dataset using the [Ultralytics](https://github.com/ultralytics/ultralytics) and [Roboflow](https://roboflow.com) platforms.

## 📁 Notebook Overview

This project guides you through:

- Checking GPU availability
- Installing YOLOv8 and dependencies
- Performing inference using a pre-trained model
- Accessing datasets from Roboflow Universe
- Training YOLOv8 on a custom dataset
- Validating and running inference on custom-trained models
- Deploying your model using Roboflow API

## 📦 Dependencies

Install the required packages:

```bash
pip install ultralytics==8.0.196 roboflow

Dataset:
his example uses a public Roboflow dataset: vehicles-q0x2v. Replace this with your own dataset by:

Uploading images to Roboflow

Exporting in YOLOv5 PyTorch format

Using the provided API to load the dataset into the notebook

Training:
We use the yolov8s.pt base model and train for 25 epochs on 800x800 image resolution.

yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=25 imgsz=800 plots=True

Results :
Training output will include:

results.png: loss and mAP curves

confusion_matrix.png: performance visualization

val_batch0_pred.jpg: predictions on validation set

best.pt: best model weights
**Run inference on images or videos:**
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=/path/to/image_or_video save=True


**You can deploy the trained model to the cloud using Roboflow**:
project.version(dataset.version).deploy(model_type="yolov8", model_path="runs/detect/train/")

**final Note:**
This notebook gives a complete pipeline to train, evaluate, and deploy a YOLOv8 object detection model on a custom dataset.
