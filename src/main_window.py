import customtkinter as ctk
import tkinter as tk
from PIL import Image
import os
from settings import *
from functions import *

class Main_Window(ctk.CTk):
    def __init__(self):
        super().__init__()

        # root settings
        # self.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}')
        self.geometry('1000x760')
        self.resizable(False, False)
        self.title('YOLO Object Detector')
        self.iconbitmap('../img/icon.ico')

        # attributes
        self.images = get_image_list(PATH_TO_IMAGES)
        self.models = get_model_list(PATH_TO_MODELS)
        self.selected_image = None
        self.selected_model = None
        self.mode = tk.IntVar(master=self, value=1) # 1 - single image, 2 - video

        # widgets
        self.image = ctk.CTkImage(light_image=Image.open(f'../img/photo.png'), # place holder image
                                    dark_image=Image.open(f'../img/photo.png'),
                                    size=(960,540))
        self.image_label = ctk.CTkLabel(self, text='', image=self.image)

        self.predict_btn = ctk.CTkButton(master=self, text='Predict',
                                       font=ctk.CTkFont(size=30),
                                       command=self.predict_btn_onclick)
        
        self.exit_btn = ctk.CTkButton(master=self, text='Exit',
                                       font=ctk.CTkFont(size=30),
                                       command=self.exit_btn_onclick)
        
        self.image_cmbox = ctk.CTkComboBox(master=self, values=self.images,
                                      state='readonly',
                                      command=self.image_cmbox_callback,
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
        
        self.image_cmbox_label = ctk.CTkLabel(master=self, text='Select Image:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.model_cmbox_label = ctk.CTkLabel(master=self, text='Select Model:',
                                   font=ctk.CTkFont(size=15, weight='bold'),
                                   height=10)
        
        self.radiobtn_image = ctk.CTkRadioButton(master=self, text="Single image", 
                                                 variable=self.mode, value=1,
                                                 command=self.radiobtn_callback, 
                                                 font=ctk.CTkFont(size=15, weight='bold'),
                                                height=10)
        self.radiobtn_video = ctk.CTkRadioButton(master=self, text="Video", 
                                                 variable=self.mode, value=2,
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
        self.exit_btn.grid(row=2, column=3, columnspan=1, padx=10, pady=10)
        self.image_cmbox.grid(row=2, column=0, columnspan=1, padx=10, pady=10, sticky='ew')
        self.model_cmbox.grid(row=4, column=0, columnspan=1, padx=10, pady=10, sticky='ew')
        self.image_cmbox_label.grid(row=1, column=0, columnspan=1, padx=20, pady=10, sticky='sw')
        self.model_cmbox_label.grid(row=3, column=0, columnspan=1, padx=20, pady=10, sticky='sw')
        self.radiobtn_image.grid(row=1, column=1, columnspan=1, padx=10, pady=10, sticky='nsew')
        self.radiobtn_video.grid(row=2, column=1, columnspan=1, padx=10, pady=10, sticky='nsew')
        

    # actions
    def predict_btn_onclick(self) -> None:
        model = load_model(f'{PATH_TO_MODELS}/{self.selected_model}')
        image = get_prediction_image(model, f'{PATH_TO_IMAGES}/{self.selected_image}')
        self.update_image(image)

    def exit_btn_onclick(self) -> None:
        self.destroy()

    def image_cmbox_callback(self, selected) -> None:
        self.selected_image = selected
        self.update_image(Image.open(f'{PATH_TO_IMAGES}/{selected}'))

    def model_cmbox_callback(self, selected) -> None:
        self.selected_model = selected

    def update_image(self, image) -> None:
        self.image = ctk.CTkImage(light_image=image,
                                    dark_image=image,
                                    size=(960,540))
        self.image_label.configure(image=self.image)
        self.image_label._image = self.image
    
    def radiobtn_callback(self):
        pass