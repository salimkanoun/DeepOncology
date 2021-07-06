import csv 
import json 
import os
from sklearn.model_selection import train_test_split 


class JSON_TO_CSV : 
    """class to prepare a csv with study uid, path, and labels from a json results from LABEL STUDIO 
    """

    def __init__(self, json_path):
        self.json_path = json_path 
        self.dataset = self.extract_result()


    def extract_result(self):
        dataset = []
        with open(self.json_path) as json_file : 
            reader = json.load(json_file)
            for info in reader :
                dataset.append(info)

        return dataset 


    def result_csv(self, image_directory,  csv_directory): 
        """[summary]

        Args:
            image_png_directory ([str]): [path of directory where the 2d_image are. model =  image_png_directory+'/'+study_folder+'/'+image_à_annoter]
            csv_directory_directory ([str]): [path of directory to save csv_file/dataset ]

        Returns:
            [type]: [description]
        """
        data = []
        error = []
        for image in self.dataset : 
            try : 
                subliste = []
                study_uid = image['data']['image'].split('-')[1].split('_')[0]
                subliste.append(study_uid)

                result_1 = image['completions'][0]['result'][0]
                result_2 = image['completions'][0]['result'][1]
                result_3 = image['completions'][0]['result'][2]
                result_4 = image['completions'][0]['result'][3]
                subliste.append(result_1)
                subliste.append(result_2)
                subliste.append(result_3)
                subliste.append(result_4)

                data.append(subliste)
            except Exception : 
                error.append(image)
        maj_data = []
        for image in data : 
            liste = []
            liste.append(image[0])
            subliste = []

            for i in range(1, 5): 
                subliste.append(image[i]['from_name'])
                subliste.append(image[i]['value']['choices'][0])
            #upperLimit 
            ind = subliste.index('upperLimit')
            liste.append(subliste[ind+1])

            #lowerLimit
            ind = subliste.index('lowerLimit')
            liste.append(subliste[ind+1])

            #rightArm
            ind = subliste.index('rightArm')
            liste.append(subliste[ind+1])

            #leftArm

            ind = subliste.index('leftArm')
            liste.append(subliste[ind+1])

            maj_data.append(liste)
        
        folder = os.listdir(image_directory) 
        
        for image in maj_data : 
            study_uid = image[0]
            for files in folder : 
                if study_uid in files : 
                    maj_data.append(files)


        with open(os.path.join(csv_directory, 'classification_dataset.csv'), 'w') as csv_file : 
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["STUDY UID", "IMAGE", "UPPER_LIMIT", "LOWER_LIMIT", "RIGHT_ARM", "LEFT_ARM"])
            for serie in maj_data : 
                csv_writer.writerow([serie[0], serie[-1], serie[1], serie[2], serie[3], serie[4]])

        self.csv_result_path = os.path.join(csv_directory, 'classification_dataset.csv')
        return self.csv_result_path