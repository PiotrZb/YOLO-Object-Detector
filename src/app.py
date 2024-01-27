import customtkinter as ctk
from main_window import Main_Window

class App:
    def __init__(self):
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('dark-blue')

        self.main_window = Main_Window()
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        self.main_window.finish_threads()
        self.main_window.destroy()

    def run(self):
        self.main_window.mainloop()