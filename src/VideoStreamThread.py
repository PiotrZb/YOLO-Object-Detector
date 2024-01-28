import threading as th
import cv2 as cv
import time
from settings import PATH_TO_VIDEOS


class VideoStreamThread(th.Thread):
    def __init__(self, selected_video, lock):
        super(VideoStreamThread, self).__init__()
        self.loop = True
        self.image = None
        self.selected_video = selected_video
        self.wait = False
        self.lock = lock
        self.daemon = True

    def pause(self):
        with self.lock:
            self.wait = True
    
    def unpause(self):
        with self.lock:
            self.wait = False

    def get_image(self):
        with self.lock:
            return self.image

    def stop(self):
        with self.lock:
            self.loop = False

    def run(self):
        video_capture = cv.VideoCapture(f'{PATH_TO_VIDEOS}\\{self.selected_video}') # video capture init
        lag = 0
        try:
            if video_capture.isOpened():

                original_fps = video_capture.get(cv.CAP_PROP_FPS) # frame rate for original video
                if original_fps < 0.5:
                    original_fps = 30

                original_time = (1 / original_fps) # single frame time

                while self.loop:
                    if self.wait:
                        time.sleep(0.5)
                        continue

                    start_time = time.time()

                    ret, self.image = video_capture.read() # read next frame

                    if not ret: # end ov video (finish thread)
                        break

                    # set const frame rate
                    diff = original_time - (time.time() - start_time)
                    if lag > 0:
                        lag -= diff
                    elif diff > 0.0:
                        time.sleep(diff)
                    else:
                        lag += diff

        finally:
            video_capture.release()