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



def random_colors(N, bright=True, seed_color=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.Random(seed_color).shuffle(colors)
    return colors


def get_src_image(pet_array, axis):
    # Background image
    v_max = np.max([np.max(pet_array), 1.0]) * 0.40
    image = pet_array.copy()
    image[image > v_max] = v_max
    image = (255 * image / v_max).astype(int)
    image = np.max(image, axis=axis)  # MIP
    # convert to rgb
    image = image[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    return image


def get_bbox(mask):
    """
    mask : shape (x, y, n_object)
    return :
        bbox: shape (n_object, 4)
    """
    bbox = []
    for i in range(mask.shape[2]):
        indexes = np.where(mask[:, :, i] == 1)
        if len(indexes[0]) == 0:
            y1, y2 = 0, 0
            x1, x2 = 0, 0
        else:
            y1, y2 = min(indexes[0]), max(indexes[0])
            x1, x2 = min(indexes[1]), max(indexes[1])
        bbox.append([y1, x1, y2, x2])

    return np.array(bbox)


def plot_diff(pet_array, gt_mask_array, pred_mask_array, axis=1, ax=None):
    """
    pet_array : shape (z, y, x)
    gt_mask_array: shape (z, y, x)
    pred_mask_array: shape (z, y, x)
    """

    # flip for correct to be at the top of the image
    pet_array = np.flip(pet_array, axis=0)
    gt_mask_array = np.flip(gt_mask_array, axis=0)
    pred_mask_array = np.flip(pred_mask_array, axis=0)

    # background of the plot
    image = get_src_image(pet_array, axis)

    #
    gt_mask_array = np.expand_dims(gt_mask_array, axis=-1)
    pred_mask_array = np.expand_dims(pred_mask_array, axis=-1)

    # mip
    gt_mask_array = np.max(gt_mask_array, axis=axis)
    pred_mask_array = np.max(pred_mask_array, axis=axis)

    plot_difference(image, gt_mask_array, pred_mask_array, ax=ax)


def plot_difference(image, gt_mask, pred_mask, ax=None):
    """
    :param image: RGB background image, ndarray of shape (x, y, 3)
    :param gt_mask: ground-truth image, ndarray of shape (x, y, 1)
    :param pred_mask: pred image, ndarray of shape (x, y, 1)
    :param ax:
    """

    gt_bbox = get_bbox(gt_mask)
    pred_bbox = get_bbox(pred_mask)

    # generate things
    gt_colors = [(0, 1, 0, .8)] * gt_mask.shape[2]  # Ground truth = green.
    gt_class_ids = np.ones(gt_mask.shape[2], dtype=int)

    pred_colors = [(1, 0, 0, 1)] * pred_mask.shape[2]  # Predictions = red
    pred_class_ids = np.ones(pred_mask.shape[2], dtype=int)

    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_ids, pred_class_ids])
    class_names = ["", ""]
    colors = gt_colors + pred_colors
    boxes = np.concatenate([gt_bbox, pred_bbox])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)

    title = "Ground Truth and Detections\n GT=green, pred=red"

    # Display
    visualize.display_instances(
        image,
        boxes, masks, class_ids,
        class_names, ax=ax,
        show_bbox=False, show_mask=True,
        colors=colors,
        title=title)


def plot_seg(pet_array, mask_array_true, mask_array_pred, axis=1, suptitle=''):
    figsize = (16, 16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(suptitle)

    #
    pet_array = pet_array * 25.0

    # flip axial axis
    pet_array = np.flip(pet_array, axis=0)
    mask_array_true = np.flip(mask_array_true, axis=0)
    mask_array_pred = np.flip(mask_array_pred, axis=0)

    # MIP
    mip_mask_true = np.max(mask_array_true, axis=axis)
    mip_mask_pred = np.max(mask_array_pred, axis=axis)

    image = get_src_image(pet_array, axis)

    # add dims
    mip_mask_true = np.expand_dims(mip_mask_true, axis=-1)
    mip_mask_pred = np.expand_dims(mip_mask_pred, axis=-1)

    bbox = get_bbox(mip_mask_true)
    colors = random_colors(bbox.shape[0])

    # plot the result for ground-truth
    class_ids, class_names = np.ones(bbox.shape[0], dtype=int), ["", ""]  # ['background', 'lymphoma']

    visualize.display_instances(image, bbox, mip_mask_true, class_ids, class_names, show_bbox=False,
                                colors=colors, ax=ax1)
    # plot the result for pred
    bbox = get_bbox(mip_mask_pred)
    colors = random_colors(bbox.shape[0])

    visualize.display_instances(image, bbox, mip_mask_pred, class_ids, class_names, show_bbox=False,
                                colors=colors, ax=ax2)

    if axis == 1:
        ax1.set_title('coronal \ntrue')
        ax2.set_title('coronal \npred')
    elif axis == 2:
        ax1.set_title('sagittal \ntrue')
        ax2.set_title('sagittal \npred')
    elif axis == 3:
        ax1.set_title('axial \ntrue')
        ax2.set_title('axial \npred')
    else:
        ax1.set_title('true')
        ax2.set_title('pred')

    plt.show()


def display_instance(pet_array, mask_array=None, axis=1, ax=None):
    if mask_array is None:
        mask_array = np.zeros(pet_array.shape)
    if ax is None:
        figsize = (16, 16)
        fig, ax = plt.subplots(figsize=figsize)

    # MIP
    image = get_src_image(pet_array, axis)
    mip_mask = np.max(mask_array, axis=axis)

    # add dims
    mip_mask = np.expand_dims(mip_mask, axis=-1)

    bbox = get_bbox(mip_mask)
    colors = random_colors(bbox.shape[0])

    # plot the result for ground-truth
    class_ids, class_names = np.ones(bbox.shape[0], dtype=int), ["", ""]

    visualize.display_instances(image, bbox, mip_mask, class_ids, class_names, show_bbox=False,
                                colors=colors, ax=ax)

    return ax


def display_roi(pet_array, roi_array, mask_array=None, axis=1, ax=None):
    """
    :param pet_array: 3d-arrray (z, y, x)
    :param roi_array: 4d-array, (n_obj, z, y, x)
    :param mask_array: 4d-array, (n_obj, z, y, x) or None
    :param axis, MIP axis
    """
    if mask_array is None:
        # mask_array = np.zeros(roi_array.shape)
        mask_array = roi_array
    if ax is None:
        figsize = (16, 16)
        fig, ax = plt.subplots(figsize=figsize)

    # (n_obj, z, y, x) to (z, y, x, n_obj)
    roi_array = np.transpose(roi_array, [1, 2, 3, 0])
    mask_array = np.transpose(mask_array, [1, 2, 3, 0])

    # head up
    pet_array = np.flip(pet_array, axis=0)
    roi_array = np.flip(roi_array, axis=0)
    mask_array = np.flip(mask_array, axis=0)

    # MIP
    image = get_src_image(pet_array, axis)
    mip_roi = np.max(roi_array, axis=axis)
    mip_mask = np.max(mask_array, axis=axis)

    # Generate some additional info
    bbox = get_bbox(mip_roi)
    colors = random_colors(bbox.shape[0])
    class_ids, class_names = np.arange(0, bbox.shape[0], dtype=int), [str(i) for i in range(bbox.shape[0])]
    # class_ids, class_names = np.ones(bbox.shape[0], dtype=int), ["", ""]

    # plot it
    visualize.display_instances(image, bbox, mip_mask, class_ids, class_names, show_bbox=True,
                                colors=colors, ax=ax)


def gif(fnames, duration, output_file):
    images = []
    for filename in fnames:
        images.append(imageio.imread(filename))

    imageio.mimsave(output_file, images, duration=duration)


def generate_gif(filename, pet_array, mask_array=None, duration=0.1, number_of_img=60):
    """
    1 GIF = 360° of PET scan with segmentation if provided

    :param filename: path/to/where/to/save/somename.gif
    :param pet_array: ndarray
    :param mask_array: ndarray. If Set to None, no mask will be plotted
    :param duration: duration of the GIF
    :param number_of_img: number of img
    """
    # Create temp dir
    dir = os.path.join(os.path.dirname(filename), 'tmp')
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create and save MIP of each rotation
    filenames_frame = []
    for i, angle in enumerate(tqdm(np.linspace(0, 360, number_of_img, endpoint=False))):
        # rotate img
        img_rotated = scipy.ndimage.interpolation.rotate(pet_array, angle, reshape=False, axes=(2, 1))
        if mask_array is not None:
            mask_rotated = scipy.ndimage.interpolation.rotate(mask_array, angle, reshape=False, axes=(2, 1))
        else:
            mask_rotated = None

        figsize = (16, 16)
        fig, ax = plt.subplots(figsize=figsize)

        display_instance(img_rotated, mask_array=mask_rotated, axis=1, ax=ax)

        fname = os.path.join(dir, filename + '_' + str(int(angle)) + '.png')
        fig.savefig(fname, bbox_inches='tight')
        filenames_frame.append(fname)

    # Create GIF from saved png
    gif(filenames_frame, duration, filename)

