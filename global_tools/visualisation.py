import SimpleITK as sitk
import sys
import numpy as np
import imageio
import scipy.ndimage
import matplotlib.pyplot as plt
from os.path import basename, splitext
import re


def create_gif(filenames, duration, path_gif):
    """
        From a list of images, generates gif
    """
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    name = splitext(basename(filenames[0][:-2]))[0]
    output_file = path_gif + '/GIF_' + name + '.gif'
    imageio.mimsave(output_file, images, duration=duration)

    return None


def create_MIP_projection(filenames, path_gif, borne_max=1.0):
    """
        From a NIFTI file filename, generates rotation MIP img .jpg
        and generates gif associated
    """

    duration = 0.1
    number_of_img = 60
    angle_filenames = []

    for filename in filenames:

        print("\nGeneration gif patient: %s" % basename(filename))

        raw_filename = splitext(basename(filename))[0]
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))

        for i, angle in enumerate(np.linspace(0, 360, number_of_img)):
            # definition of loading bar
            length = round((i + 1) / number_of_img * 30)
            loading_bar = "[" + "=" * length + ">" + "-" * (30 - length) + "]"
            sys.stdout.write("\r%s/%s %s" % (str(i + 1), str(number_of_img), loading_bar))

            vol_angle = scipy.ndimage.interpolation.rotate(img, angle, reshape=False, axes=(2, 1))
            MIP = np.amax(vol_angle, axis=1)

            f = plt.figure(figsize=(10, 10))
            axes = plt.gca()
            plt.imshow(MIP, cmap='Greys', origin='lower', vmax=borne_max)
            axes.set_axis_off()
            angle_filename = path_gif + '/' + raw_filename + "." + str(int(angle)) + ".png"
            angle_filenames.append(angle_filename)
            f.savefig(angle_filename, bbox_inches='tight')

            plt.close()

        create_gif(angle_filenames, duration, path_gif)

    return None

def plot_MIP_pdf(PET_scan, CT_scan, Mask):
    study_uid = re.sub('_nifti_PT\.nii(\.gz)?', '', os.path.basename((pet_path)))

    # for TEP visualisation
    PET_scan = np.where(PET_scan > 1.0, 1.0, PET_scan)
    PET_scan = np.where(PET_scan < 0.0, 0.0, PET_scan)

    # for CT visualisation
    CT_scan = np.where(CT_scan > 1.0, 1.0, CT_scan)
    CT_scan = np.where(CT_scan < 0.0, 0.0, CT_scan)

    # # for correct visualisation
    # PET_scan = np.flip(PET_scan, axis=0)
    # CT_scan = np.flip(CT_scan, axis=0)
    # Mask = np.flip(Mask, axis=0)

    # stacked projections
    PET_scan = np.hstack((np.amax(PET_scan, axis=1), np.amax(PET_scan, axis=2)))
    CT_scan = np.hstack((np.amax(CT_scan, axis=1), np.amax(CT_scan, axis=2)))
    Mask = np.hstack((np.amax(Mask, axis=1), np.amax(Mask, axis=2)))

    # Plot
    f = plt.figure(figsize=(15, 10))
    f.suptitle(study_uid, fontsize=15)
    # f.suptitle('splitext(basename(PET_id))[0)', fontsize=15)

    plt.subplot(121)
    plt.imshow(CT_scan, cmap=color_CT, origin='lower')
    plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
    plt.axis('off')
    plt.title('PET/CT', fontsize=20)

    plt.subplot(122)
    plt.imshow(CT_scan, cmap=color_CT, origin='lower')
    plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
    plt.imshow(np.where(Mask, 0, np.nan), cmap=color_MASK, origin='lower')
    plt.axis('off')
    plt.title('PET/CT + Segmentation', fontsize=20)

    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()