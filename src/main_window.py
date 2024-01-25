import customtkinter as ctk
import tkinter as tk
from PIL import Image
from settings import *
from functions import *
from ultralytics import YOLO
import cv2 as cv
import threading as th
import time
import numpy as np


global_lock = th.Lock()


class Main_Window(ctk.CTk):
    def __init__(self):
        super().__init__()

        # root settings
        # self.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}')
        self.geometry('1000x600')
        self.resizable(False, False)
        self.title('YOLO Object Detector')
        self.iconbitmap('../img/icon.ico')

        # attributes
        self.images = get_image_list(PATH_TO_IMAGES)
        self.models = get_model_list(PATH_TO_MODELS)
        self.videos = get_video_list(PATH_TO_VIDEOS)
        self.selected_image = None
        self.selected_video = None
        self.selected_model = None
        self.mode = tk.IntVar(master=self, value=1) # 1 - single image, 2 - video
        self.model = None
        self.current_stream_frame = None
        self.stream_active = False
        self.video_prediction_active = False
        self.video_predict_thread = None
        self.stream_thread = None
        self.single_image_thread = None
        self.pause= False
        self.camera_active = False
        self.camera_thread = None

        # widgets
        self.image = ctk.CTkImage(light_image=Image.open(f'../img/photo.png'), # place holder image
                                    dark_image=Image.open(f'../img/photo.png'),
                                    size=(640,384))
        
        self.image_label = ctk.CTkLabel(self, text='', image=self.image)

        self.predict_btn = ctk.CTkButton(master=self, text='Predict',
                                       font=ctk.CTkFont(size=15),
                                       command=self.predict_btn_onclick,
                                       state=ctk.DISABLED)
        
        self.exit_btn = ctk.CTkButton(master=self, text='Exit',
                                       font=ctk.CTkFont(size=15),
                                       command=self.exit_btn_onclick)
        
        self.reply_btn = ctk.CTkButton(master=self, text='Reply',
                                       font=ctk.CTkFont(size=15),
                                       command=self.reply_btn_onclick,
                                       state=ctk.DISABLED)

        self.source_cmbox = ctk.CTkComboBox(master=self, values=self.images,
                                      state='readonly',
                                      command=self.source_cmbox_callback,
                                      width=300,
                                      height=30,
                                      font=ctk.CTkFont(size=15, weight='bold'),
                                      dropdown_font=ctk.CTkFont(size=15, weight='bold'))
        
        self.model_cmbox = ctk.CTkComboBox(master=self, values=self.models,
                                      state='readonly',
                                      command=self.model_cmbox_callback,
                                      width=300, 
                                      height=30,
                                      font=ctk.CTkFont(size=15, weight='bold'),
                                      dropdown_font=ctk.CTkFont(size=15, weight='bold'))
        
        self.source_cmbox_label = ctk.CTkLabel(master=self, text='Select Source:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.mode_label = ctk.CTkLabel(master=self, text='Select Mode:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.model_cmbox_label = ctk.CTkLabel(master=self, text='Select Model:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.radiobtn_image = ctk.CTkRadioButton(master=self, text="Single Image", 
                                                 variable=self.mode, value=1,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        
        self.radiobtn_video = ctk.CTkRadioButton(master=self, text="Video", 
                                                 variable=self.mode, value=2,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        
        self.radiobtn_camera = ctk.CTkRadioButton(master=self, text="Camera", 
                                                 variable=self.mode, value=3,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)

        # grid deffinition
        self.rowconfigure(0, weight=10)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=1)
        self.columnconfigure(2,weight=1)
        self.columnconfigure(3,weight=1)

        # layout
        self.image_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky='enw')
        self.predict_btn.grid(row=2, column=2, columnspan=1, padx=10, pady=10)
        self.reply_btn.grid(row=2, column=3, columnspan=1, padx=10, pady=10)
        self.exit_btn.grid(row=4, column=3, columnspan=1, padx=10, pady=10)
        self.source_cmbox.grid(row=2, column=0, columnspan=1, padx=10, pady=10, sticky='ew')
        self.model_cmbox.grid(row=4, column=0, columnspan=1, padx=10, pady=10, sticky='ew')
        self.source_cmbox_label.grid(row=1, column=0, columnspan=1, padx=20, pady=10, sticky='sw')
        self.model_cmbox_label.grid(row=3, column=0, columnspan=1, padx=20, pady=10, sticky='sw')
        self.mode_label.grid(row=1, column=1, columnspan=1, padx=20, pady=10, sticky='sw')
        self.radiobtn_image.grid(row=2, column=1, columnspan=1, padx=10, pady=10, sticky='nsew')
        self.radiobtn_video.grid(row=3, column=1, columnspan=1, padx=10, pady=10, sticky='nsew')
        self.radiobtn_camera.grid(row=4, column=1, columnspan=1, padx=10, pady=10, sticky='nsew')
        

    # actions
    def start_threads(self):
        self.stream_active = True
        self.stream_thread = th.Thread(target=self.start_video_stream)
        self.stream_thread.start()
        self.video_prediction_active = True
        self.video_predict_thread = th.Thread(target=self.predict_on_video)
        self.video_predict_thread.start()
        self.predict_btn.configure(text="Pause")

    def predict_btn_onclick(self) -> None:
        if self.mode.get() == 1:
            self.single_image_thread = th.Thread(target=self.predict_on_single_image)
            self.single_image_thread.start()
        elif self.mode.get() == 2 and not self.video_prediction_active:
            self.start_threads()
        elif self.pause:
            self.pause = False
            self.predict_btn.configure(text="Pause")
            self.reply_btn.configure(state=ctk.DISABLED)
            print("123123")
        else:
            self.pause = True
            self.predict_btn.configure(text="Resume")
            self.reply_btn.configure(state=ctk.NORMAL)


    def exit_btn_onclick(self) -> None:
        self.finish_threads()
        self.destroy()


    def reply_btn_onclick(self) -> None:
        self.finish_threads()
        self.start_threads()
        self.pause = False
        self.reply_btn.configure(state=ctk.DISABLED)
    

    def finish_threads(self):
        if self.stream_thread is not None:
            self.stream_active = False
            self.stream_thread.join()
        if self.video_predict_thread is not None:
            self.video_prediction_active = False
            self.video_predict_thread.join()
        if self.camera_thread is not None:
            self.camera_active = False
            self.camera_thread.join()


    def source_cmbox_callback(self, selected) -> None:
        if self.mode.get() == 1:
            self.selected_image = selected
            self.update_image(Image.open(f'{PATH_TO_IMAGES}/{selected}'))
        elif self.mode.get() == 2:
            self.selected_video = selected
            # TODO

        if self.model is not None:
            self.predict_btn.configure(state=ctk.NORMAL)


    def model_cmbox_callback(self, selected) -> None:
        self.selected_model = selected
        self.model = YOLO(f'{PATH_TO_MODELS}/{self.selected_model}')

        if (self.selected_image is not None and self.mode.get() == 1) or (self.selected_video is not None and self.mode.get() == 2):
            self.predict_btn.configure(state=ctk.NORMAL)


    def update_image(self, image) -> None:
        self.image = ctk.CTkImage(light_image=image,
                                    dark_image=image,
                                    size=(640,384))
        self.image_label.configure(image=self.image)
        self.image_label.image = self.image
    

    def radiobtn_callback(self):
        if self.mode.get() == 1:
            self.finish_threads()
            self.source_cmbox.configure(values=self.images)
        elif self.mode.get() == 2:
            self.finish_threads()
            self.source_cmbox.configure(values=self.videos)
        elif self.mode.get() == 3:
            self.finish_threads()
            self.camera_active = True
            if self.camera_thread is None:
                self.camera_thread = th.Thread(target=self.start_camera_stream)
            self.camera_thread.start()
            
        self.source_cmbox.set('')
        self.update_image(Image.open(f'../img/photo.png'))
        self.selected_image = None
        self.selected_video = None
        self.pause = False
        self.reply_btn.configure(state=ctk.DISABLED)
        self.predict_btn.configure(state=ctk.DISABLED, text="Predict")
    

    # thread functions
    def predict_on_single_image(self):
        with global_lock:
            results = self.model.predict(f'{PATH_TO_IMAGES}/{self.selected_image}', iou=0.5)
        image = Image.fromarray(results[0].plot()[..., ::-1])
        with global_lock:
            self.update_image(image)


    def start_video_stream(self):
        video_capture = cv.VideoCapture(f'{PATH_TO_VIDEOS}/{self.selected_video}') # video capture init
        lag = 0

        if video_capture.isOpened():

            original_fps = video_capture.get(cv.CAP_PROP_FPS) # frame rate for original video
            if original_fps < 0.5:
                original_fps = 30

            original_time = (1 / original_fps) # single frame time

            while self.stream_active:
                if self.pause:
                    time.sleep(0.5)
                    continue

                start_time = time.time()

                with global_lock: # save acces
                    ret, self.current_stream_frame = video_capture.read() # read next frame

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
            
        self.video_prediction_active = False
        video_capture.release()


    def predict_on_video(self):

        # thread loop
        while self.video_prediction_active: 

            if self.pause:
                    time.sleep(0.5)
                    continue

            if self.current_stream_frame is not None:
                start_time = time.time()
                
                with global_lock: # save acces
                    frame_copy = self.current_stream_frame.copy()
                
                # predict
                results = self.model(frame_copy)
                
                prediction_image = np.ascontiguousarray(results[0].plot()[..., ::-1], dtype=np.uint8)
                
                # print fps
                cv.putText(prediction_image, f"FPS {(1/(time.time() - start_time)):.1f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # update image
                if self.video_prediction_active:
                    self.update_image(Image.fromarray(prediction_image))

    def start_camera_stream(self):
        video_capture = cv.VideoCapture(0) # video capture init

        if video_capture.isOpened():

            while self.camera_active:

                start_time = time.time()

                with global_lock: # save acces
                    ret, frame = video_capture.read() # read next frame

                if not ret: # end ov video (finish thread)
                    break
                
                with global_lock: # save acces
                    frame_copy = frame.copy()
                
                # predict
                results = self.model(frame_copy)
                
                prediction_image = np.ascontiguousarray(results[0].plot()[..., ::-1], dtype=np.uint8)
                
                # print fps
                cv.putText(prediction_image, f"FPS {(1/(time.time() - start_time)):.1f}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # update image
                if self.camera_active:
                    self.update_image(Image.fromarray(prediction_image))
            
        video_capture.release()