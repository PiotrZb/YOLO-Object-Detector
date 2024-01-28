import os
import sys
from settings import *


# @author: Nautilius
# @link: https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

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