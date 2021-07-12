import SimpleITK as sitk 
import matplotlib.pyplot as plt 
from dicom_to_cnn.tools.pre_processing.threshold_mask import * 
from dicom_to_cnn.model.post_processing.clustering.Watershed import Watershed
from losses.Metrics import * 
import numpy as np
from skimage import filters

def get_array_from_nifti(nifti_path): 
    img = sitk.ReadImage(nifti_path)
    spacing = img.GetSpacing()

    array = sitk.GetArrayFromImage(img)
    return array, spacing

def applied_watershed_on_inference(mask_nifti_path, pet_nifti_path):
    mask_img = sitk.ReadImage(mask_nifti_path)
    pet_img = sitk.ReadImage(pet_nifti_path)
    model = Watershed(mask_img, pet_img)
    ws_img = model.applied_watershed_model()
    return ws_img

def thresh_mask(mask_array, pet_array, threshold):
    new_mask = np.zeros(mask_array.shape[1:], dtype=np.uint8)
    for num_slice in range(mask_array.shape[0]):
        mask_slice = mask_array[num_slice]  # R.O.I
        roi = pet_array[mask_slice > 0]
        if len(roi) == 0:
            continue
        try:
            # apply threshold
            if threshold == 'otsu' : 
                t = filters.threshold_otsu(roi)
            elif threshold == '0.41' : 
                t = np.max(roi) * float(threshold)
            elif threshold == '2.5' or threshold == '4.0' : 
                t = float(threshold)
            new_mask[np.where((pet_array >= t) & (mask_slice > 0))] = 1

        except Exception as err : 
            pass

    return new_mask.astype('uint8')

def multi_seuil_mask(mask_array, pet_array):
    # get 3D meta information
    if len(mask_array.shape) == 3:
        mask_array = np.expand_dims(mask_array, axis=0)
    else:
        mask_array = np.transpose(mask_array, (3,0,1,2))

    new_masks = []
        #otsu 
        #print('otsu')
    new_masks.append(thresh_mask(mask_array, pet_array, threshold='otsu'))
        #print('41%')
    new_masks.append(thresh_mask(mask_array, pet_array, threshold='0.41'))
        #2.5
        #print('2.5')
    new_masks.append(thresh_mask(mask_array, pet_array, threshold='2.5'))
        #4.0
        #print('4.0')
    new_masks.append(thresh_mask(mask_array, pet_array, threshold='4.0'))
    new_mask = np.stack(new_masks, axis=3)
    new_mask = np.mean(new_mask, axis=3)
    return new_mask


def calcul_dice_global(pred_array, true_array):
    pred_array = np.expand_dims(pred_array, axis=-1)
    pred_array = np.expand_dims(pred_array, axis=0)

    true_array = np.round(true_array)
    if len(true_array.shape) == 4 : 
        true_array = np.amax(true_array, axis=-1)
    #true_array[np.where(true_array != 0)] = 1
    true_array = np.expand_dims(true_array, axis=-1)
    true_array = np.expand_dims(true_array, axis=0)
    dice = metric_dice(true_array, pred_array, axis=(1, 2, 3, 4))
    return dice


def calcul_dice_threshold(pred_array, true_array, pet_array, thresh = 0.41):
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
    if len(true_array.shape) == 4 : 
        true_array = np.sum(true_array, axis=-1)
    
    #true_array = np.round(true_array)
    true_array[np.where(true_array != 0)] = 1
    true_array = np.expand_dims(true_array, axis=-1)
    true_array = np.expand_dims(true_array, axis=0)
    dice = metric_dice(true_array, pred_array, axis=(1, 2, 3, 4))
    
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



 