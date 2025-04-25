!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.0.196

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

!yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True
Image(filename='runs/detect/predict/dog.jpeg', height=600)

model = YOLO(f'{HOME}/yolov8n.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)

results[0].boxes.xyxy
results[0].boxes.conf
results[0].boxes.cls

!mkdir {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="XRQVccQMwquXx3LwrIGc")
project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
version = project.version(2)
dataset = version.download("yolov8")

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
!ls {HOME}/runs/detect/train/
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source=/content/testsample.mp4 save=True

from google.colab import drive
drive.mount('/content/drive')

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:30]:
    display(Image(filename=image_path, width=600))
    print("\n")

project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")
model = project.version(dataset.version).model

import os, random
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)

pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred
