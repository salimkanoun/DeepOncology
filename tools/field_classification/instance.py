import numpy as np 
from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 

def generate_instance_from_nifti(img_dict):
    """prepare a batch for classification model from a dict

    Args:
        img_dict ([dict]): [{ct_img : value(a nifti image.nii), upper_limit : value, lower_limit : value, right_arm : value, left_arm : value}]

    Returns:
        [tuple]: [return the 2D MIP of the CT image and its encoded label]
    """
    resampled_array = Nifti(img_dict['ct_img']).resample(shape_matrix=(256, 256, 1024), shape_physic=(700, 700, 2000))
    resampled_array[np.where(resampled_array < 500)] = 0 #500 UH
    normalize = resampled_array[:,:,:,]/np.max(resampled_array)
    mip_generator = MIP_Generator(normalize)
    mip = mip_generator.project(angle=0)
    mip = np.expand_dims(mip, -1)
    label = encoding_instance(img_dict)

    return mip, label

def encoding_instance(dictionnary):
    """encoding label with integer

    Args:
        dictionnary ([dict]): ['right arm': value, 'left_arm':value, 'upper_limit':value, 'lower_limit':value, other values allowed]

    Returns:
        [list]: [return a list with encoded integer labels in order ['head', 'legs', 'right_arm', 'left_arm'] ]
    """
    label = []
    
        #upper Limit 
    if dictionnary['upper_limit'] == 'Vertex' : 
        label.append(0)
    if dictionnary['upper_limit'] == 'Eye'  or dictionnary['upper_limit'] == 'Mouth' : 
        label.append(1)

        #lower Limit
    if dictionnary['lower_limit'] == 'Hips' : 
        label.append(0)
    if dictionnary['lower_limit'] == 'Knee': 
        label.append(1)
    if dictionnary['lower_limit'] == 'Foot':
        label.append(2)

        #right Arm 
    if dictionnary['right_arm'] == 'down' : 
        label.append(0)
    if dictionnary['right_arm'] == 'up' : 
        label.append(1)

        #left Arm 
    if dictionnary['left_arm'] == 'down' : 
        label.append(0)
    if dictionnary['left_arm'] == 'up' : 
        label.append(1)

    return label