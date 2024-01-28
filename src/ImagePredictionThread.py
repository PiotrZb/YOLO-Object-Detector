from settings import PATH_TO_IMAGES
from PIL import Image
import customtkinter as ctk
import threading as th


class ImagePredictionThread(th.Thread):
    def __init__(self, model=None, iou=0.5, conf=0.25, image_label=None, selected_image=None, image_to_save=None):
        super(ImagePredictionThread, self).__init__()
        self.model = model
        self.iou = iou
        self.conf = conf
        self.image_label = image_label
        self.selected_image = selected_image
        self.daemon = True
        self.image_to_save = image_to_save

    def run(self):
        if self.model is not None:
            results = self.model.predict(f'{PATH_TO_IMAGES}/{self.selected_image}', iou=self.iou, conf=self.conf)
            image = Image.fromarray(results[0].plot()[..., ::-1])

            if self.image_label is not None:
                img = ctk.CTkImage(light_image=image,
                                    dark_image=image,
                                    size=(640,384))
                self.image_label.configure(image=img)
                self.image_label.image = img

                self.image_to_save["image"] = image