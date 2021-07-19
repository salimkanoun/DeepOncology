import numpy as np 
import SimpleITK as sitk 
import csv
import pandas as pd 


class CSV_TO_ARRAYS:
    """class to convert csv info into list of array for classification algorithm 
    """

    def __init__(self, csv_path): 
        """ csv contains path CT scans 
        """
        self.csv_path = csv_path
        self.dataset = self.extract_dataset()
    
    def extract_dataset(self):
        with open(self.csv_path, 'r') as csv_file :
            reader = csv.reader(csv_file, delimiter = ',') 
            dataset = []
            for row in reader :
                dataset.append(row)
                
        del dataset[0] #delete first lign
        return dataset
    