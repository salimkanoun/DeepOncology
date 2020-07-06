import SimpleITK as sitk
import sys
import numpy as np
import imageio
import scipy.ndimage
import matplotlib.pyplot as plt
from os.path import basename, splitext


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