
import numpy as np

from mrcnn import visualize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import colorsys
import random


def random_colors(N, bright=True, seed=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.Random(seed).shuffle(colors)
    return colors


def plot_tmtv(df, show=True):
    x_key = 'tmtv_true'
    y_key = 'tmtv_pred'

    xlim = 0.0, df[[x_key, y_key]].max().max()

    g = sns.jointplot(x_key, y_key,
                      data=df,
                      xlim=xlim, ylim=xlim,
                      kind="reg")

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')

    r2 = r2_score(df['tmtv_true'], df['tmtv_pred'])
    corr, _ = pearsonr(df['tmtv_true'], df['tmtv_pred'])
    title = 'r2 = {:.2f}\nPearson Corr = {}'.format(r2, corr)
    if show:
        plt.show()
    else:
        return g


def plot_seg(pet_array, mask_array, ax, axis=1):
    # figsize = (16, 16)
    # fig, axes = plt.subplots(1, 3, figsize=figsize)

    # # flip axial axis
    # pet_array = np.flip(pet_array_, axis=0)
    # mask_array_true = np.flip(mask_array_, axis=0)

    # MIP
    mip_pet = np.max(pet_array, axis=axis)
    mip_mask = np.max(mask_array, axis=axis)

    # prepapre img for plotting
    mip_pet[mip_pet > 5.0] = 5.0
    mip_pet = (255 * mip_pet / 5.0).astype(int)
    # convert to RBG + MIP
    image = mip_pet[:, :, None] * np.ones(3, dtype=int)[None, None, :]

    # generate bounding box from the segmentation
    bbox = []
    for i in range(mip_mask.shape[2]):
        indexes = np.where(mip_mask[:, :, i])
        y1, y2 = min(indexes[0]), max(indexes[0])
        x1, x2 = min(indexes[1]), max(indexes[1])
        bbox.append([y1, x1, y2, x2])

    bbox = np.array(bbox)

    # labels
    colors = random_colors(1) #random_colors(bbox.shape[0])
    class_ids, class_names = np.ones(bbox.shape[0], dtype=int), ["", ""]  # ['background', 'lymphoma']

    # plot it
    visualize.display_instances(image, bbox, mip_mask, class_ids, class_names, show_bbox=False,
                                colors=colors, ax=ax)

    return ax









