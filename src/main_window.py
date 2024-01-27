import customtkinter as ctk
import tkinter as tk
from PIL import Image
from settings import *
from functions import *
from ultralytics import YOLO
import threading as th
import numpy as np
from CameraThread import CameraThread
from VideoStreamThread import VideoStreamThread
from ImagePredictionThread import ImagePredictionThread
from VideoPredictionThread import VideoPredictionThread


# global variables
global_lock = th.Lock()


class Main_Window(ctk.CTk):
    def __init__(self):
        super().__init__()

        # root settings
        # self.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}')
        self.geometry(f'{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}')
        self.resizable(False, False)
        self.title('YOLO Detektor')
        self.iconbitmap('../img/icon.ico')

        # attributes
        self.images = get_image_list(PATH_TO_IMAGES)
        self.models = get_model_list(PATH_TO_MODELS)
        self.videos = get_video_list(PATH_TO_VIDEOS)
        self.selected_image = None
        self.selected_video = None
        self.selected_model = None
        self.mode = tk.IntVar(master=self, value=1) # 1 - single image, 2 - video, 3 - camera
        self.model = None
        self.video_predict_thread = None
        self.video_stream_thread = None
        self.single_image_thread = None
        self.camera_thread = None
        self.conf = tk.IntVar(master=self, value=25)
        self.iou = tk.DoubleVar(master=self, value=0.5)
        self.conf_text = tk.StringVar(master=self, value="25 %")
        self.iou_text = tk.StringVar(master=self, value="0.50")
        self.pause = False

        # widgets
        self.image = ctk.CTkImage(light_image=Image.open(f'../img/photo.png'), # place holder image
                                    dark_image=Image.open(f'../img/photo.png'),
                                    size=(640,384))
        
        self.movie_icon = ctk.CTkImage(light_image=Image.open(f'../img/movie.png'), # place holder image
                                    dark_image=Image.open(f'../img/movie.png'),
                                    size=(24,24))
        
        self.camera_icon = ctk.CTkImage(light_image=Image.open(f'../img/camera.png'), # place holder image
                                    dark_image=Image.open(f'../img/camera.png'),
                                    size=(24,24))
        
        self.image_icon = ctk.CTkImage(light_image=Image.open(f'../img/image.png'), # place holder image
                                    dark_image=Image.open(f'../img/image.png'),
                                    size=(24,24))
        
        self.exit_icon = ctk.CTkImage(light_image=Image.open(f'../img/exit.png'), # place holder image
                                    dark_image=Image.open(f'../img/exit.png'),
                                    size=(24,24))
        
        self.save_icon = ctk.CTkImage(light_image=Image.open(f'../img/save.png'), # place holder image
                                    dark_image=Image.open(f'../img/save.png'),
                                    size=(24,24))
        
        self.replay_icon = ctk.CTkImage(light_image=Image.open(f'../img/replay.png'), # place holder image
                                    dark_image=Image.open(f'../img/replay.png'),
                                    size=(24,24))
        
        self.play_icon = ctk.CTkImage(light_image=Image.open(f'../img/play.png'), # place holder image
                                    dark_image=Image.open(f'../img/play.png'),
                                    size=(24,24))
        
        self.pause_icon = ctk.CTkImage(light_image=Image.open(f'../img/pause.png'), # place holder image
                                    dark_image=Image.open(f'../img/pause.png'),
                                    size=(24,24))
        
        self.folder_icon = ctk.CTkImage(light_image=Image.open(f'../img/folder.png'), # place holder image
                                    dark_image=Image.open(f'../img/folder.png'),
                                    size=(24,24))
        
        self.ai_icon = ctk.CTkImage(light_image=Image.open(f'../img/ai.png'), # place holder image
                                    dark_image=Image.open(f'../img/ai.png'),
                                    size=(24,24))
        
        self.image_label = ctk.CTkLabel(master=self, text='', image=self.image)

        self.movie_icon_label = ctk.CTkLabel(master=self, text='', image=self.movie_icon)

        self.camera_icon_label = ctk.CTkLabel(master=self, text='', image=self.camera_icon)

        self.image_icon_label = ctk.CTkLabel(master=self, text='', image=self.image_icon)

        self.predict_btn = ctk.CTkButton(master=self, text='Predykcja',
                                       font=ctk.CTkFont(size=15),
                                       command=self.predict_btn_onclick,
                                       state=ctk.DISABLED,
                                       image=self.play_icon,
                                       compound="right")
        
        self.exit_btn = ctk.CTkButton(master=self, text='Wyjście',
                                       font=ctk.CTkFont(size=15),
                                       command=self.exit_btn_onclick,
                                       image=self.exit_icon,
                                       compound="right")
        
        self.save_btn = ctk.CTkButton(master=self, text='Zapisz',
                                       font=ctk.CTkFont(size=15),
                                       command=self.save_btn_onclick,
                                       image=self.save_icon,
                                       compound="right")
        
        self.reply_btn = ctk.CTkButton(master=self, text='Ponów',
                                       font=ctk.CTkFont(size=15),
                                       command=self.reply_btn_onclick,
                                       state=ctk.DISABLED,
                                       image=self.replay_icon,
                                       compound="right")

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
        
        self.source_cmbox_label = ctk.CTkLabel(master=self, text='   Wybierz źródło:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10,
                                   image=self.folder_icon,
                                   compound="left")
        
        self.mode_label = ctk.CTkLabel(master=self, text='Wybierz tryb:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.model_cmbox_label = ctk.CTkLabel(master=self, text='   Wybierz model:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10,
                                   image=self.ai_icon,
                                   compound="left")
        
        self.radiobtn_image = ctk.CTkRadioButton(master=self, text="Obraz", 
                                                 variable=self.mode, value=1,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        
        self.radiobtn_video = ctk.CTkRadioButton(master=self, text="Wideo", 
                                                 variable=self.mode, value=2,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        
        self.radiobtn_camera = ctk.CTkRadioButton(master=self, text="Kamera", 
                                                 variable=self.mode, value=3,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        
        self.iou_slider = ctk.CTkSlider(master=self, from_=0, to=1, 
                                        command=self.iou_slider_event, 
                                        number_of_steps=20,
                                        orientation="vertical")
        
        self.conf_slider = ctk.CTkSlider(master=self, from_=0, to=100, 
                                        command=self.conf_slider_event, 
                                        number_of_steps=20,
                                        orientation="vertical")
        
        self.iou_label = ctk.CTkLabel(master=self, text='IoU',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.conf_label = ctk.CTkLabel(master=self, text='Pewność',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.iou_value_label = ctk.CTkLabel(master=self, textvariable=self.iou_text,
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.conf_value_label = ctk.CTkLabel(master=self, textvariable=self.conf_text,
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)

        # grid deffinition
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=10)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=1)
        
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=0)
        self.columnconfigure(2,weight=1)
        self.columnconfigure(3,weight=1)
        self.columnconfigure(4,weight=1)

        # layout
        self.image_label.grid(row=0, column=0, columnspan=3, rowspan=3, padx=10, pady=10, sticky='enw')
        self.conf_slider.grid(row=1, column=3, columnspan=1, rowspan=1, padx=10, pady=10)
        self.conf_label.grid(row=0, column=3, columnspan=1, rowspan=1, padx=10, pady=10)
        self.conf_value_label.grid(row=2, column=3, columnspan=1, rowspan=1, padx=10, pady=10)
        self.iou_slider.grid(row=1, column=4, columnspan=1, rowspan=1, padx=10, pady=10)
        self.iou_label.grid(row=0, column=4, columnspan=1, rowspan=1, padx=10, pady=10)
        self.iou_value_label.grid(row=2, column=4, columnspan=1, rowspan=1, padx=10, pady=10)
        self.predict_btn.grid(row=4, column=3, columnspan=1, rowspan=1, padx=10, pady=10)
        self.reply_btn.grid(row=4, column=4, columnspan=1, rowspan=1, padx=10, pady=10)
        self.exit_btn.grid(row=6, column=4, columnspan=1, rowspan=1, padx=10, pady=10)
        self.save_btn.grid(row=6, column=3, columnspan=1, rowspan=1, padx=10, pady=10)
        self.source_cmbox.grid(row=4, column=0, columnspan=1, rowspan=1, padx=10, pady=10, sticky='ew')
        self.model_cmbox.grid(row=6, column=0, columnspan=1, rowspan=1, padx=10, pady=10, sticky='ew')
        self.source_cmbox_label.grid(row=3, column=0, columnspan=1, rowspan=1, padx=20, pady=10, sticky='sw')
        self.model_cmbox_label.grid(row=5, column=0, columnspan=1, rowspan=1, padx=20, pady=10, sticky='sw')
        self.mode_label.grid(row=3, column=1, columnspan=1, padx=20, rowspan=1, pady=10, sticky='sw')
        self.radiobtn_image.grid(row=4, column=1, columnspan=1, rowspan=1, padx=10, pady=10, sticky='nsew')
        self.radiobtn_video.grid(row=5, column=1, columnspan=1, rowspan=1, padx=10, pady=10, sticky='nsew')
        self.radiobtn_camera.grid(row=6, column=1, columnspan=1, rowspan=1, padx=10, pady=10, sticky='nsew')
        self.image_icon_label.grid(row=4, column=2, columnspan=1, rowspan=1, padx=10, pady=10, sticky='w')
        self.movie_icon_label.grid(row=5, column=2, columnspan=1, rowspan=1, padx=10, pady=10, sticky='w')
        self.camera_icon_label.grid(row=6, column=2, columnspan=1, rowspan=1, padx=10, pady=10, sticky='w')
        


# --------------------------------------------------------- Methods ---------------------------------------------------------



    def iou_slider_event(self, value):
        self.iou_text.set(f"{round(value, 2):.2f}")


    def conf_slider_event(self, value):
        self.conf_text.set(f"{int(value)} %")


    def exit_btn_onclick(self) -> None:
        self.finish_threads()
        self.destroy()

    def save_btn_onclick(self) -> None:
        pass


    def finish_threads(self):
        if self.video_stream_thread is not None and self.video_stream_thread.is_alive():
            self.video_stream_thread.stop()
            #self.video_stream_thread.join()
            self.stream_thread = None

        if self.video_predict_thread is not None and self.video_predict_thread.is_alive():
            self.video_predict_thread.stop()
            #self.video_predict_thread.join()
            self.video_predict_thread = None

        if self.camera_thread is not None and self.camera_thread.is_alive():             
            self.camera_thread.stop()
            #self.camera_thread.join()
            self.camera_thread = None

        if self.single_image_thread is not None and self.single_image_thread.is_alive():
            self.single_image_thread.join()
            self.single_image_thread = None

    
    def model_cmbox_callback(self, selected) -> None:
        self.selected_model = selected
        self.model = YOLO(f'{PATH_TO_MODELS}/{self.selected_model}')

        if (self.selected_image is not None and self.mode.get() == 1) or (self.selected_video is not None and self.mode.get() == 2):
            self.predict_btn.configure(state=ctk.NORMAL)


    def start_video_threads(self):
        self.video_stream_thread = VideoStreamThread(self.selected_video, global_lock)
        self.video_stream_thread.start()
        self.video_predict_thread = VideoPredictionThread(self.model, self.iou.get(), self.conf.get() / 100, self.image_label, self.video_stream_thread, global_lock)
        self.video_predict_thread.start()


    def reply_btn_onclick(self) -> None:
        self.finish_threads()
        self.start_video_threads()


    def update_image(self, image):
        img = ctk.CTkImage(light_image=image,
                            dark_image=image,
                            size=(640,384))
        self.image_label.configure(image=img)
        self.image_label.image = img


    def source_cmbox_callback(self, selected) -> None:
        if self.mode.get() == 1:
            self.selected_image = selected
            self.update_image(Image.open(f'{PATH_TO_IMAGES}/{selected}'))
        elif self.mode.get() == 2:
            self.selected_video = selected
        if self.model is not None:
            self.predict_btn.configure(state=ctk.NORMAL)


    def predict_btn_onclick(self) -> None:
        if self.mode.get() == 1:
            self.single_image_thread = ImagePredictionThread(self.model, self.iou.get(), self.conf.get() / 100, self.image_label, self.selected_image)
            self.single_image_thread.start()

        elif self.mode.get() == 2 and self.video_predict_thread is None:
            self.start_video_threads()
            self.radiobtn_camera.configure(state=ctk.DISABLED)
            self.radiobtn_video.configure(state=ctk.DISABLED)
            self.radiobtn_image.configure(state=ctk.DISABLED)
            self.source_cmbox.configure(state=ctk.DISABLED)
            self.iou_slider.configure(state=ctk.DISABLED)
            self.conf_slider.configure(state=ctk.DISABLED)

        elif self.pause:
            self.pause = False
            self.video_stream_thread.unpause()
            self.video_predict_thread.unpause()
            self.predict_btn.configure(text="Pauza")
            self.reply_btn.configure(state=ctk.DISABLED)
            self.radiobtn_camera.configure(state=ctk.DISABLED)
            self.radiobtn_video.configure(state=ctk.DISABLED)
            self.radiobtn_image.configure(state=ctk.DISABLED)
            self.source_cmbox.configure(state=ctk.DISABLED)
            self.iou_slider.configure(state=ctk.DISABLED)
            self.conf_slider.configure(state=ctk.DISABLED)

        else:
            self.pause = True
            self.video_stream_thread.pause()
            self.video_predict_thread.pause()
            self.predict_btn.configure(text="Wznów")
            self.reply_btn.configure(state=ctk.NORMAL)
            self.radiobtn_camera.configure(state=ctk.NORMAL)
            self.radiobtn_video.configure(state=ctk.NORMAL)
            self.radiobtn_image.configure(state=ctk.NORMAL)
            self.source_cmbox.configure(state=ctk.NORMAL)
            self.iou_slider.configure(state=ctk.NORMAL)
            self.conf_slider.configure(state=ctk.NORMAL)


    def radiobtn_callback(self):
        self.finish_threads()
        self.source_cmbox.configure(state=ctk.NORMAL)

        if self.mode.get() == 1:
            self.source_cmbox.configure(values=self.images)

        elif self.mode.get() == 2:
            self.source_cmbox.configure(values=self.videos)

        elif self.mode.get() == 3:
            self.camera_thread = CameraThread(self.model, self.iou.get(), self.conf.get() / 100, self.image_label)
            self.camera_thread.start()
            self.source_cmbox.configure(state=ctk.DISABLED)
            
        self.source_cmbox.set('')
        self.update_image(Image.open(f'../img/photo.png'))
        self.selected_image = None
        self.selected_video = None
        self.reply_btn.configure(state=ctk.DISABLED)
        self.predict_btn.configure(state=ctk.DISABLED, text="Predykcja")