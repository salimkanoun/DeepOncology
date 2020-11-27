import denseCRF3D
import numpy as np


class DenseCRF(object):
    """
    https://github.com/HiLab-git/SimpleCRF/blob/master/examples/demo_densecrf3d.py
    """

    def __init__(self, keys=('image', 'label'), dense_crf_param=None,
                 norm_image=False, new_key='label'):
        self.keys = keys
        self.norm_image = norm_image
        self.new_key = new_key

        if dense_crf_param is None:
            dense_crf_param = {'MaxIterations': 10,
                               'PosW': 3.0,
                               'PosRStd': 5,
                               'PosCStd': 5,
                               'PosZStd': 5,
                               'BilateralW': 3.0,
                               'BilateralRStd': 5.0,
                               'BilateralCStd': 5.0,
                               'BilateralZStd': 5.0,
                               'ModalityNum': 2,
                               'BilateralModsStds': (7.0, 50.0)}

        self.dense_crf_param = dense_crf_param

    def __call__(self, img_dict):
        image = img_dict[self.keys[0]]
        label = img_dict[self.keys[1]]

        if self.norm_image:
            image = self.normalize(image)
        image[image < 0] = 0
        image[image > 1] = 1
        image = np.asarray(image * 255, np.uint8)

        P = label
        P = np.asarray([1.0 - P, P], np.float32)
        P = np.transpose(P, [1, 2, 3, 0])

        img_dict[self.new_key] = self.apply_CRF3D(image, P)

        return img_dict

    def normalize(self, image):
        a_min, a_max = np.min(image, axis=(0, 1, 2)), np.max(image, axis=(0, 1, 2))
        image = (image - a_min) / (a_max - a_min)

        return image

    def apply_CRF3D(self, image, probs):
        """
        input parameters:
            I: a numpy array of shape [D, H, W, C], where C is the channel number
               type of I should be np.uint8, and the values are in [0, 255]
            P: a probability map of shape [D, H, W, L], where L is the number of classes
               type of P should be np.float32
            param: a tuple giving parameters of CRF. see the following two examples for details.
        """

        return denseCRF3D.densecrf3d(image, probs, self.dense_crf_param)


class DenseCRFbias(object):
    """
    https://github.com/HiLab-git/SimpleCRF/blob/master/examples/demo_densecrf3d.py
    """

    def __init__(self, keys=('image', 'probs', 'bias'), dense_crf_param=None, ratio=0.5,
                 norm_image=False, new_key='label'):
        self.keys = keys
        self.norm_image = norm_image
        self.new_key = new_key

        if dense_crf_param is None:
            dense_crf_param = {'MaxIterations': 10,
                               'PosW': 3.0,
                               'PosRStd': 5,
                               'PosCStd': 5,
                               'PosZStd': 5,
                               'BilateralW': 3.0,
                               'BilateralRStd': 5.0,
                               'BilateralCStd': 5.0,
                               'BilateralZStd': 5.0,
                               'ModalityNum': 2,
                               'BilateralModsStds': (7.0, 50.0)}

        self.dense_crf_param = dense_crf_param
        self.ratio = ratio

    def __call__(self, img_dict):
        image = img_dict[self.keys[0]]
        label = img_dict[self.keys[1]]
        bias = img_dict[self.keys[2]]

        if self.norm_image:
            image = self.normalize(image)
        image[image < 0] = 0
        image[image > 1] = 1
        image = np.asarray(image * 255, np.uint8)

        P = label
        # P = np.where(P > 0.0, ratio, 0.0) + (1 - ratio) * P
        P = self.ratio * bias + (1 - self.ratio) * P
        P = np.asarray([1.0 - P, P], np.float32)
        P = np.transpose(P, [1, 2, 3, 0])

        img_dict[self.new_key] = self.apply_CRF3D(image, P)

        return img_dict

    def normalize(self, image):
        a_min, a_max = np.min(image, axis=(0, 1, 2)), np.max(image, axis=(0, 1, 2))
        image = (image - a_min) / (a_max - a_min)

        return image

    def apply_CRF3D(self, image, probs):
        """
        input parameters:
            I: a numpy array of shape [D, H, W, C], where C is the channel number
               type of I should be np.uint8, and the values are in [0, 255]
            P: a probability map of shape [D, H, W, L], where L is the number of classes
               type of P should be np.float32
            param: a tuple giving parameters of CRF. see the following two examples for details.
        """

        return denseCRF3D.densecrf3d(image, probs, self.dense_crf_param)





