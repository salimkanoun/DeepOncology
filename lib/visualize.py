import colorsys
import random
import numpy as np
import SimpleITK as sitk

from mrcnn import visualize

import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_seg(pet_array, mask_array_true, mask_array_pred, axis=1):
    figsize = (16, 16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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


# def plot_roi(pet_img, roi_img, mask_img=None):
#     if mask_img is None:
#         mask_img = np.zeros(roi_img.shape)
#     pass


