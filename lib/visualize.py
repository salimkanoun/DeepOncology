import os
from tqdm import tqdm

import colorsys
import random
import numpy as np
# import SimpleITK as sitk
import scipy
import imageio
from fpdf import FPDF

from mrcnn import visualize

# import seaborn as sns
import matplotlib.pyplot as plt



def inference_pet_projection(pet_array, inference_array, study_uid, patient_id, study, axis, directory , vmin, vmax):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 15))
    fig.suptitle("patient_id = {} \n study_uid : {} \n study : {}".format(patient_id, study_uid, study))


    MIP_pet = np.amax(pet_array, axis = axis)
    ax1.imshow(MIP_pet, cmap = 'gray', vmin = 0, vmax = 10, origin='lower')
    ax1.set_title('PET scan')
    ax1.axis('off')


    MIP_inf = np.amax(inference_array,axis=axis)
    plt.imshow(MIP_pet, cmap = 'gray', vmin = 0, vmax = 10, origin='lower')
    plt.imshow(np.where(MIP_inf, 0, np.nan), cmap='Set1', alpha = 0.5, origin='lower')
    ax2.set_title('prediction')
    ax2.axis('off')
    

    filename = os.path.join(directory, study_uid+'_mip_inference_'+str(axis)+'.jpg')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

def generate_inference_pet_projection_pdf(liste_paths_images, directory, pdf_filename):
    pdf = FPDF()
    for mip in liste_paths_images : 
        pdf.add_page()
        pdf.image(mip, w=190, h = 230)
        os.remove(mip)
    pdf.output(os.path.join(directory, pdf_filename))

    return None 

