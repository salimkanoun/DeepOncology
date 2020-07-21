import SimpleITK as sitk
import sys
import numpy as np
import imageio
import scipy.ndimage
import matplotlib.pyplot as plt
import re
import os


def get_study_uid(img_path):
    return re.sub('_nifti_(PT|mask|CT)\.nii(\.gz)?', '', os.path.basename(img_path))


def create_gif(filenames, duration, output_file):
    """
    :param filenames: list, path to images
    :param duration: int, duration of the gif
    :param output_file:, str, path/to/name.gif
    """
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave(output_file, images, duration=duration)


def create_MIP_projection(img, filename_gif, mask=None, vmax=1.0, duration=0.1, number_of_img=60):
    """
    Create a gif of rotating 3D img

    :param img: image, simple itk image or path/to/image.nii
    :param mask: image, simple itk image, or path/to/image.nii
    :param filename_gif: path/to/file.gif
    :param vmax: max value for plotting
    :param duration: duration of the gif in seconds ?
    :param number_of_img: number images of the gif i.e number of rotation of the img
    """
    if isinstance(img, str):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img))
    if mask is not None:
        if isinstance(img, str):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask))

    path_gif = os.path.dirname(filename_gif)
    if not os.path.exists(os.path.join(path_gif, 'temp')):
        os.makedirs(os.path.join(path_gif, 'temp'))

    angle_filenames = []

    for i, angle in enumerate(np.linspace(0, 360, number_of_img)):
        # definition of loading bar
        length = round((i + 1) / number_of_img * 30)
        loading_bar = "[" + "=" * length + ">" + "-" * (30 - length) + "]"
        sys.stdout.write("\r%s/%s %s" % (str(i + 1), str(number_of_img), loading_bar))

        # rotate img
        vol_angle = scipy.ndimage.interpolation.rotate(img, angle, reshape=False, axes=(2, 1))

        # calculate MIP
        MIP = np.max(vol_angle, axis=1)

        # plot MIP
        f = plt.figure(figsize=(10, 10))
        axes = plt.gca()
        plt.imshow(MIP, cmap='Greys', origin='lower', vmax=vmax)
        if mask is not None:
            vol_angle_mask = scipy.ndimage.interpolation.rotate(mask, angle, reshape=False, axes=(2, 1))
            MIP_mask = np.max(vol_angle_mask, axis=1)
            plt.imshow(MIP_mask, cmap='jet', alpha=0.5, origin='lower')
        axes.set_axis_off()
        angle_filename = os.path.join(path_gif, 'temp', os.path.basename(filename_gif) + "_" + str(int(angle)) + ".png")
        angle_filenames.append(angle_filename)
        f.savefig(angle_filename, bbox_inches='tight')

        plt.close()

    create_gif(angle_filenames, duration, path_gif)

#
# def plot_MIP_pdf(PET_scan, CT_scan, Mask):
#     study_uid = re.sub('_nifti_PT\.nii(\.gz)?', '', os.path.basename((pet_path)))
#
#     # for TEP visualisation
#     PET_scan = np.where(PET_scan > 1.0, 1.0, PET_scan)
#     PET_scan = np.where(PET_scan < 0.0, 0.0, PET_scan)
#
#     # for CT visualisation
#     CT_scan = np.where(CT_scan > 1.0, 1.0, CT_scan)
#     CT_scan = np.where(CT_scan < 0.0, 0.0, CT_scan)
#
#     # # for correct visualisation
#     # PET_scan = np.flip(PET_scan, axis=0)
#     # CT_scan = np.flip(CT_scan, axis=0)
#     # Mask = np.flip(Mask, axis=0)
#
#     # stacked projections
#     PET_scan = np.hstack((np.amax(PET_scan, axis=1), np.amax(PET_scan, axis=2)))
#     CT_scan = np.hstack((np.amax(CT_scan, axis=1), np.amax(CT_scan, axis=2)))
#     Mask = np.hstack((np.amax(Mask, axis=1), np.amax(Mask, axis=2)))
#
#     # Plot
#     f = plt.figure(figsize=(15, 10))
#     f.suptitle(study_uid, fontsize=15)
#     # f.suptitle('splitext(basename(PET_id))[0)', fontsize=15)
#
#     plt.subplot(121)
#     plt.imshow(CT_scan, cmap=color_CT, origin='lower')
#     plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
#     plt.axis('off')
#     plt.title('PET/CT', fontsize=20)
#
#     plt.subplot(122)
#     plt.imshow(CT_scan, cmap=color_CT, origin='lower')
#     plt.imshow(PET_scan, cmap=color_PET, alpha=transparency, origin='lower')
#     plt.imshow(np.where(Mask, 0, np.nan), cmap=color_MASK, origin='lower')
#     plt.axis('off')
#     plt.title('PET/CT + Segmentation', fontsize=20)
#
#     pdf.savefig()  # saves the current figure into a pdf page
#     plt.close()
