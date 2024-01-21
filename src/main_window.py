import customtkinter as ctk
import tkinter as tk
from PIL import Image


class Main_Window(ctk.CTk):
    def __init__(self):
        super().__init__()

        #self.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}')
        self.geometry('1000x700')
        self.resizable(False, False)
        self.title('YOLO Object Detector')
        self.iconbitmap('../img/icon.ico')

        # widgets
        image = ctk.CTkImage(light_image=Image.open('../data/test.png'), dark_image=Image.open('../data/test.png'), size=(960,540))
        image_label = ctk.CTkLabel(self, text='', image=image)
        image_label.pack(pady=10)

        # layout

        # actions
