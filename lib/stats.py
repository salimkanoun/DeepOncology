import SimpleITK as sitk 
import matplotlib.pyplot as plt 

from library_dicom.dicom_processor.tools.threshold_mask import * 
from library_dicom.post_processing.WatershedModel import WatershedModel
from losses.Metrics import * 


def get_array_from_nifti(nifti_path): 
    img = sitk.ReadImage(nifti_path)
    spacing = img.GetSpacing()

    array = sitk.GetArrayFromImage(img)
    return array, spacing

def applied_watershed_on_inference(mask_nifti_path, pet_nifti_path):
    model = WatershedModel(mask_nifti_path, pet_nifti_path, type = '3d')
    ws_array, label_number = model.watershed_model(0.5)
    return ws_array 

def applied_threshold_on_matrix(mask_array, pet_array, thresh = 0.41):
    """threshold mask 

    Args:
        mask_array ([ndarray]): [(z,x,y,c) or (z,x,y)]
        pet_array ([ndarray]): [(z,x,y)]
        thresh (float, optional): [description]. Defaults to 0.41.
    """
    if len(mask_array.shape) == 3 : 
        number_of_roi = 1 
    else : 
        number_of_roi = mask_array.shape[3]
    for i in range(number_of_roi):
        if len(mask_array.shape) == 3 : 
            roi = mask_array
        else : 
            roi = mask_array[:,:,:,i]
        roi_copy = np.copy(roi)
        pet_copy = np.copy(pet_array)
        pet_copy[roi == 0] = 0 
        if thresh < 1.0 : 
            seuil = thresh * np.max(pet_copy)
        else : seuil = thresh 
        roi_copy[np.where(pet_copy < seuil)] = 0

        if len(mask_array.shape) == 3 : 
            mask_array = roi_copy 
        else : 
            mask_array[:,:,:,i] = roi_copy 
    return mask_array



def calcul_dice(pred_array, true_array, pet_array, thresh = 0.41):
    """[summary]

    Args:
        pred_array ([array]): [(z,x,y)]
        true_array ([type]): [(z,x,y,c)]
        thresh (float, optional): [description]. Defaults to 0.41.
    """

    pred_array = threshold_matrix(pred_array, pet_array, thresh)
    pred_array[np.where(pred_array != 0)] = 1
   
    pred_array = np.expand_dims(pred_array, axis=-1)
    pred_array = np.expand_dims(pred_array, axis=0)
    
    true_array = threshold_matrix(true_array, pet_array, thresh)
    #true_array = applied_threshold_on_matrix(true_array, pet_array, thresh = thresh)
    if len(true_array.shape) == 4 : 
        true_array = np.sum(true_array, axis=-1)
    
    true_array[np.where(true_array != 0)] = 1
    true_array = np.expand_dims(true_array, axis=-1)
    true_array = np.expand_dims(true_array, axis=0)


    dice = metric_dice(true_array, pred_array, axis=(1, 2, 3, 4))
    """
    true_array=np.sum(true_array, axis=-1)
    true_array=np.sum(true_array, axis=0)
    pred_array=np.sum(pred_array, axis=-1)
    pred_array=np.sum(pred_array, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 15))
    axis = 2
    MIP_pet = np.amax(pet_array, axis = axis)
    MIP_true = np.amax(true_array,axis=axis) 
    MIP_pred = np.amax(pred_array,axis=axis)
    ax1.imshow(MIP_pet, cmap = 'gray', vmin = 0, vmax = 10, origin='lower')
    ax1.imshow(np.where(MIP_true, 0, np.nan), cmap='Set1', alpha = 0.5, origin='lower')
    ax1.axis('off')

    ax2.imshow(MIP_pet, cmap = 'gray', vmin = 0, vmax = 10, origin='lower')
    ax2.imshow(np.where(MIP_pred, 0, np.nan), cmap='Set1', alpha = 0.5, origin='lower')
        
    ax2.axis('off')
    plt.show()
    
    """
    return dice 

def calcul_tmtv(pred_array, true_array, pet_array, spacing, thresh = 0.41):
    pred_array = threshold_matrix(pred_array, pet_array, thresh)
    pred_array[np.where(pred_array != 0)] = 1

    z,x,y = np.where(pred_array != 0)
    number_of_pixel = len(z)
    volume_voxel = float(spacing[0])*float(spacing[1])*float(spacing[2])
    tmtv_pred = number_of_pixel * volume_voxel * 1e-3 #mm3
    
   
    #true_array = threshold_matrix(true_array, pet_array, thresh)
    true_array = applied_threshold_on_matrix(true_array, pet_array, thresh = thresh)
    if len(true_array.shape) == 4 : 
        true_array = np.sum(true_array, axis=-1)
    true_array[np.where(true_array != 0)] = 1
    z,x,y = np.where(true_array != 0)
    number_of_pixel = len(z)
    tmtv_true = number_of_pixel * volume_voxel * 1e-3 #mm3

    return tmtv_pred, tmtv_true 

def calcul_mean_on_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.mean(np.array(list_tmtv_pred)), np.mean(np.array(list_tmtv_true))

def calcul_median_on_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.median(np.array(list_tmtv_pred)), np.median(np.array(list_tmtv_true))

def calcul_min_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.min(np.array(list_tmtv_pred)), np.min(np.array(list_tmtv_true))

def calcul_max_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.max(np.array(list_tmtv_pred)), np.max(np.array(list_tmtv_true))

def calcul_sd_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.std(np.array(list_tmtv_pred)), np.std(np.array(list_tmtv_true))

def calcul_q1_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.quantile(np.array(list_tmtv_pred), 0.25), np.quantile(np.array(list_tmtv_true), 0.25)

def calcul_q3_tmtv(list_tmtv_pred, list_tmtv_true):
    return np.quantile(np.array(list_tmtv_pred), 0.75), np.quantile(np.array(list_tmtv_true), 0.75)



 