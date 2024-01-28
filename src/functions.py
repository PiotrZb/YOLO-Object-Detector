import os
from settings import *

def get_image_list(path):
    if os.path.exists(path) and os.path.isdir(path):
        return [x for x in os.listdir(path) if x.split('.')[-1] in SUPPORTED_IMAGES_FORMATS]
    else:
        print('Path to images is invalid.')
        return -1
    
def get_video_list(path):
    if os.path.exists(path) and os.path.isdir(path):
        return [x for x in os.listdir(path) if x.split('.')[-1] in SUPPORTED_VIDEOS_FORMATS]
    else:
        print('Path to videos is invalid.')
        return -1
    
def get_model_list(path):
    if os.path.exists(path) and os.path.isdir(path):
        return [x for x in os.listdir(path) if x.endswith('.pt')]
    else:
        print('Path to models is invalid.')
        return -1