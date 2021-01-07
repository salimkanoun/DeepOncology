import numpy as np 
import SimpleITK as sitk 
import csv
from classification.pre_process.Prep_Instance import Prep_Instance
import pandas as pd 


class Preprocessing:
    """class to prepare list of array and corresponding labels for classification algorithm 
    """

    def __init__(self, csv_path): 
        """ csv contains path png and json with annotation
        """
        self.csv_path = csv_path
        self.dataset = self.extract_dataset()


    
    def extract_dataset(self):
        with open(self.csv_path, 'r') as csv_file :
            reader = csv.reader(csv_file, delimiter = ',') #liste pour chaque ligne 
            dataset = []
            for row in reader :
                dataset.append(row)
                
        del dataset[0] #enlever première ligne

        return dataset


    def normalize_encoding_dataset(self):
        liste = []
        label = []
        #liste_test = []
        #label_test = []
        #liste_val = []
        #label_val = []
        for serie in self.dataset : 
            #if "train" in serie : 
            #sub = []
            instance_object = Prep_Instance(serie[1]) 
            instance_array = instance_object.ct_array #matrice normalisé
            #sub.append(serie[0])
            #sub.append(instance_array)
            liste.append(instance_array)
                
                #encoding
            subliste = instance_object.encoding_instance(serie)
            label.append(subliste)
            #elif "test" in serie : 
                #instance_object = Prep_Instance(serie[1]) 
                #instance_array = instance_object.ct_norm_array #matrice normalisé
                #liste_test.append(instance_array)
                
                #encoding
                #subliste = instance_object.encoding_instance(serie)
                #label_test.append(subliste)

            #else : #val
                #instance_object = Prep_Instance(serie[1]) 
                #instance_array = instance_object.ct_norm_array #matrice normalisé
                #liste_val.append(instance_array)
                
                #encoding
                #subliste = instance_object.encoding_instance(serie)
                #label_val.append(subliste)
        return np.asarray(liste), np.asarray(label)


    




