#DICOM TO CNN 
#TOOLS -> CLASSIFICATION 

import os 
import SimpleITK as sitk  
import numpy as np
import matplotlib.pyplot as plt 


"""
method to read nifti_ct, normalize, resample, get MIP and resize, and save as png or nifti 
"""

def read_nifti_ct(ct_path):
    return sitk.ReadImage(ct_path)



def normalize_CT(ct_path):
    ct_img = read_nifti_ct(ct_path)
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    #intensityWindowingFilter.SetOutputMaximum(1)
    #intensityWindowingFilter.SetOutputMinimum(0)
    windowMax = 1024
    windowMin = -1024
    intensityWindowingFilter.SetWindowMaximum(windowMax)
    intensityWindowingFilter.SetWindowMinimum(windowMin)
    return intensityWindowingFilter.Execute(ct_img)


def resample_CT(ct_img, study_uid):
    #ct_img = read_nifti_ct(ct_path)
    spacing = ct_img.GetSpacing()
    origin = ct_img.GetOrigin()

    #generate mip array
    array = sitk.GetArrayFromImage(ct_img)
    mip = np.amax(array, axis=1)
    #print("mip shape :", mip.shape)
    #plt.imshow(np.rot90(mip, k =2))
    #plt.show()

    #generate mip image 
    mip_img = sitk.GetImageFromArray(mip)
    mip_img.SetDirection((1,0,0,1))
    mip_img.SetOrigin((origin[0], origin[2]))
    mip_img.SetSpacing((spacing[0], spacing[2]))

    #target spacing, and size
    spacing_x = 700/256 #mm
    spacing_y = 2000/1024 #mm
    mip_img_size = mip_img.GetSize() #pixel
    #print("mip_img_size :", mip_img_size)
    mip_img_spacing = mip_img.GetSpacing() #mm
    #print("mip_img_spacing :", mip_img_spacing)
    mip_img_origin = mip_img.GetOrigin() #mm
    mip_img_direction = mip_img.GetDirection() 
    

    true_x = mip_img_size[0] * mip_img_spacing[0] #mm
    true_y = mip_img_size[1] * mip_img_spacing[1] #mm 
    #print("largeur :", true_x)
    #print("hauteur :", true_y)

    new_size_x = int((true_x * 256) / 700) #pixel
    new_size_y = int((true_y * 1024) / 2000) #pixel
    #print(new_size_x)
    #print(new_size_y)

    #applied transformation
    transformation = sitk.ResampleImageFilter()
    transformation.SetOutputDirection(mip_img_direction)
    transformation.SetOutputOrigin(mip_img_origin)
    transformation.SetSize((new_size_x, new_size_y))
    transformation.SetOutputSpacing((spacing_x, spacing_y))
    transformation.SetInterpolator(sitk.sitkBSpline)
    new_img = transformation.Execute(mip_img)

    result = sitk.GetArrayFromImage(new_img)
    result = np.rot90(result, k=2)
    print("result :", result.shape)
    center = [512, 127]
    x = int(result.shape[0]/2)
    y = int(result.shape[1]/2)
    #print(x)
    #print(y)
    sommet_x = center[0] - x 
    sommet_y = center[1] - y 

    #print(sommet_x)
    #print(sommet_y)
    new_array = np.zeros((1024, 256))
    if result.shape[1] != 256 : 
        new_array[sommet_x:sommet_x + result.shape[0], sommet_y:sommet_y + result.shape[1]] = result
    else : 
        new_array[sommet_x:sommet_x + result.shape[0], 0:256] = result

    #save_mip_nifti(new_array, mip_img_origin, mip_img_direction, mip_img_spacing, directory_path, study_uid)
    return new_array

def save_mip(mip, directory_path, study_uid):
    f = plt.figure(figsize=(9,9))
    axes = plt.gca()
    axes.set_axis_off()

    plt.imshow(mip, cmap = 'gray')

    filename = study_uid+'_mip_ct'+".png"
    full_filename = os.path.join(directory_path, filename)
    f.savefig(full_filename, bbox_inches='tight')
    plt.close()        
    return full_filename

def save_mip_nifti(mip, origin, direction, spacing, directory_path, study_uid):
    img = sitk.GetImageFromArray(mip)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    img.SetSpacing(spacing)
    filename = directory_path+'/'+study_uid+'_mip_ct_nifti_'+".nii"
    sitk.WriteImage(img, filename)

    return None