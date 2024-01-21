import customtkinter as ctk
from main_window import Main_Window


class App:
    def __init__(self):
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('dark-blue')

        self.main_window = Main_Window()

    def run(self):
        self.main_window.mainloop()