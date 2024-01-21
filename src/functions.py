from PIL import Image
from ultralytics import YOLO
import os
from settings import *

def get_image_list(path):
    if os.path.exists(path) and os.path.isdir(path):
        return [x for x in os.listdir(path) if x.split('.')[-1] in SUPPORTED_IMAGES_FORMATS]
    else:
        print('Path to images is invalid.')
        return -1
    
def get_model_list(path):
    if os.path.exists(path) and os.path.isdir(path):
        return [x for x in os.listdir(path) if x.endswith('.pt')]
    else:
        print('Path to models is invalid.')
        return -1
    
def load_model(path):
    return YOLO(path)

def get_prediction_image(model, path_to_image):
    results = model(path_to_image)
    image_array = results[0].plot()
    return Image.fromarray(image_array[..., ::-1])