import threading as th
import cv2 as cv
import time
import numpy as np
from PIL import Image
import customtkinter as ctk


class CameraThread(th.Thread):
    def __init__(self, model=None, iou=0.5, conf=0.25, image_label=None, image_to_save=None, lock=None):
        super(CameraThread, self).__init__()
        self.model = model
        self.iou = iou
        self.conf = conf
        self.loop = True
        self.predict_on = True
        self.image = None
        self.image_label = image_label
        self.lock = lock
        self.daemon = True
        self.image_to_save = image_to_save

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

    def start_predictions(self):
        with self.lock:
            self.predict_on = True
    
    def stop_predictions(self):
        with self.lock:
            self.predict_on = False

    def run(self):
        video_capture = cv.VideoCapture(0) # video capture init
        try:
            if video_capture.isOpened():
                while self.loop:
                    start_time = time.time() # for fps calculating

                    ret, frame = video_capture.read() # read next frame

                    if not ret: # end ov video (finish thread)
                        break

                    # get prediction
                    if self.model is not None and self.predict_on:
                        results = self.model.predict(frame, iou=self.iou, conf=self.conf)
                        prediction_image = np.ascontiguousarray(results[0].plot()[..., ::-1], dtype=np.uint8)
                    # if predictions are off, show default frame
                    else:
                        prediction_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    # show fps
                    cv.putText(prediction_image, f"FPS {(1/(time.time() - start_time)):.1f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.image = Image.fromarray(prediction_image)

                    # update frame
                    if self.image_label is not None:
                        img = ctk.CTkImage(light_image=self.image,
                                        dark_image=self.image,
                                        size=(640,384))
                        self.image_label.configure(image=img)
                        self.image_label.image = img

                        with self.lock:
                            self.image_to_save["image"] = self.image
        
        finally:
            video_capture.release()