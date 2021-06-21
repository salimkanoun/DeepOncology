import numpy as np 
import SimpleITK as sitk 
import json 
from PIL import Image 

class Instance : 
    """class to prepare an instance : transform png -> array, + encoding labels (for one image)
    """

    def __init__(self, png_img_path):
        self.png_img_path = png_img_path 
        self.image_2d = self.normalise_instance()
        self.image_2d = np.reshape(self.image_2d, (self.image_2d.shape[0], self.image_2d.shape[1], 1))

    def normalise_instance(self): 
        img = Image.open(self.png_img_path).convert('LA')
        array = np.array(img)
        array[np.where(array < 185)] = 0 #garder le squelette
        array = array[:,:,0]/255 #normalise

        return array


    def encoding_instance(self, liste):
        """encoding label 

        Args:
            liste ([list]): [study_id, png_img_path, upper_limit, lower_limit, right_arm, left_arm]

        Returns:
            [list]: [return a list with encoded labels ]
        """
        label = []
    
        #upper Limit 
        if liste[2] == 'Vertex' : 
            label.append(0)
        if liste[2] == 'Eye'  or liste[2] == 'Mouth' : 
            label.append(1)

        #lower Limit
        if liste[3] == 'Hips' : 
            label.append(0)
        if liste[3] == 'Knee': 
            label.append(1)
        if liste[3] == 'Foot':
            label.append(2)

        #right Arm 
        if liste[4] == 'down' : 
            label.append(0)
        if liste[4] == 'up' : 
            label.append(1)

        #left Arm 
        if liste[5] == 'down' : 
            label.append(0)
        if liste[5] == 'up' : 
            label.append(1)

        return label