from ultralytics import YOLO
import os
import logging
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
fn = os.path.join(ROOT_DIR, 'best.pt')

model = YOLO(fn)

def log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

def predict_img(filepath):
    log_msg('Predicting Signature')
    detections = model.predict(filepath)
    return detections

# detections = predict(r"C:\Users\360ja\Desktop\Work\signature_detection_func\static\temp\img.png")
# print(detections)