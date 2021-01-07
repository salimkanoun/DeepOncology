import numpy as np 
import SimpleITK as sitk 
import json 
from PIL import Image 

class Prep_Instance : 
    """class to prepare an instance : transform png -> array, + encoding labels (for one image)
    """

    def __init__(self, png_img_path):
        self.png_img_path = png_img_path 
        self.ct_array = self.normalise_instance()
        self.ct_array = np.reshape(self.ct_array, (self.ct_array.shape[0], self.ct_array.shape[1], 1))

    def normalise_instance(self): 
        img = Image.open(self.png_img_path).convert('LA')
        array = np.array(img)
        array[np.where(array < 185)] = 0 #garder le squelette
        array = array[:,:,0]/255 #normalise

        return array

    """
    def normalize_instance(self):
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(1)
        intensityWindowingFilter.SetOutputMinimum(0)
        new_img = intensityWindowingFilter.Execute(self.ct_img)

        return new_img 
    """

    #ici mettre methode encodage pour une instance 
    def encoding_instance(self, liste):
        label = []
    
        #upper Limit 
        if liste[2] == 'Vertex' : 
            label.append(0)
        if liste[2] == 'Eye'  or liste[2] == 'Mouth' : 
            label.append(1)

        #lower Limit
        if liste[3] == 'Hips' : 
            label.append(0)
        if liste[3] == 'Knee': # or liste[3] == 'Foot': 
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