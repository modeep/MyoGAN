import random
import cv2
import os
import PIL
from PIL import Image

import numpy as np

class DataLoader_Continous():
    def __init__(self, data_path, is_real_image, data_type, emg_length, is_flatten):
        self.data_path = data_path

    def load_image(self):
        image_path_array = (os.listdir(self.data_path))
        # Only for mac
        # image_path_array.remove('.DS_Store')
        image = Image.open(self.data_path + random.choice(image_path_array))
        image = image.resize((128, 128))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (128, 128))
        return img

    def get_images(self, num):
        images = []
        # labels = []
        
        for i in range(num):
            # image, label = self.load_image()
            image = self.load_image()
            images.append(image)
            # labels.append(label)

        # return np.array(images), np.array(labels)
        return np.array(images)
