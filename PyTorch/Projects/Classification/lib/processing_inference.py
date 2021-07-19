import numpy as np 
from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
from io import BytesIO
import torch
def preprocessing_image_to_bytes(image, output_shape, angle = 0): 
    objet = Nifti(image)
    resampled = objet.resample(shape=output_shape)
    mip_generator = MIP_Generator(resampled)
    array=mip_generator.project(angle=angle)
    array[np.where(array < 500)] = 0 #500 UH
    array[np.where(array > 1024)] = 1024 #1024 UH
    array = array[:,:,]/1024
    array = np.expand_dims(array, axis=0)
    array = array.astype(np.double)
    array = np.expand_dims(array, axis=0)
    array = torch.from_numpy(np.array(array))

    return array

def postprocessing_classification(output_model): 
    dict_labels = {'head': ['Vertex', 'Eyes/Mouth'], 'legs': ['Hips', 'Knees', 'Foot'], 'right_arm': ['down', 'up'], 'left_arm': ['up', 'down']}
    output_order = ['head', 'legs', 'right_arm', 'left_arm']
    output = list(output_model)
    result = []
    for i in range(len(output)): 
        output[i]= output[i].detach().numpy()
        result.append(np.where(output[i][0] == max(output[i][0]))[0][0])

    labelled_results = {}

    if len(dict_labels.keys())!= len(result):
        "Dimension error: Number of outputs != Number of labels"
    else: 
        for i in range(len(result)): 

            labelled_results[output_order[i]]=(dict_labels[output_order[i]][result[i]])

    return labelled_results