import threading as th
import cv2 as cv
import time
import numpy as np
from PIL import Image
import customtkinter as ctk


class VideoPredictionThread(th.Thread):
    def __init__(self, model=None, iou=0.5, conf=0.25, image_label=None, stream=None, lock=None):
        super(VideoPredictionThread, self).__init__()
        self.model = model
        self.iou = iou
        self.conf = conf
        self.loop = True
        self.predict_on = True
        self.image = None
        self.image_label = image_label
        self.wait = False
        self.lock = lock
        self.stream = stream

    def set_model(self, model):
        with self.lock:
            self.model = model
    
    def set_iou(self, value):
        with self.lock:
            self.iou = value

    def set_conf(self, value):
        with self.lock:
            self.conf = value

    def get_image(self):
        with self.lock:
            return self.image

    def stop(self):
        with self.lock:
            self.loop = False

    def pause(self):
        with self.lock:
            self.wait = True

    def unpause(self):
        with self.lock:
            self.wait = False

    def run(self):
        # thread loop
        while self.loop:
            if self.wait:
                time.sleep(0.5)
                continue
                
            if self.stream is not None:
                start_time = time.time()
                
                frame_copy = self.stream.get_image()

                if frame_copy is not None:
                
                    # predict
                    results = self.model.predict(frame_copy, iou=self.iou, conf=self.conf)
                    prediction_image = np.ascontiguousarray(results[0].plot()[..., ::-1], dtype=np.uint8)
                    
                    # print fps
                    cv.putText(prediction_image, f"FPS {(1/(time.time() - start_time)):.1f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # update image
                    self.image = Image.fromarray(prediction_image)
                    
                    if self.image_label is not None:
                        img = ctk.CTkImage(light_image=self.image,
                                                            dark_image=self.image,
                                                            size=(640,384))
                        self.image_label.configure(image=img)
                        self.image_label.image = img

                else:
                    time.sleep(0.5)

            else:
                time.sleep(0.5)