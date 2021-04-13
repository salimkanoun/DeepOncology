import tensorflow as tf
import numpy as np
import os


class DataGeneratorFromDict(tf.keras.utils.Sequence):
    """A class to create a DataGenerator object for model.fit()
    

    
    """

    def __init__(self,
                 images_paths,
                 transforms,
                 batch_size=1,
                 shuffle=True,
                 x_key='input',
                 y_key='output'):
        """
        :param images_paths: list[dict] or dict[dict]
        :param transforms: transformer to apply to data
        :param batch_size: batch size
        :param shuffle: bool. If set to true, indexes will be suffled at each end of epoch.
        :param x_key: key corresponding to input of neural network
        :param y_key: key correspond to output of neural network
        """

        self.images_paths = images_paths
        self.transforms = transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.x_key = x_key
        self.y_key = y_key

    def __len__(self):
        """
        :return: int, the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.array(list(self.images_paths.keys())) if isinstance(self.images_paths, dict) else np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: int, position of the batch in the Sequence
        :return: tuple of numpy array, (X_batch, Y_batch) of shape (batch_size, ...)
        """

        # select indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # prepare the batch
        X_batch = []
        Y_batch = []
        for idx in indexes:
            img_dict = self.images_paths[idx]
            if self.transforms is not None:
                img_dict = self.transforms(img_dict)

            # add it to the batch
            X_batch.append(img_dict[self.x_key])
            Y_batch.append(img_dict[self.y_key])

        X_batch = np.array(X_batch)
        Y_batch = np.array(Y_batch)
        return X_batch, Y_batch


class DataGenerator_3D_from_numpy(tf.keras.utils.Sequence):

    def __init__(self,
                 pp_dir,
                 subset,
                 mask_keys,
                 batch_size=1,
                 shuffle=False,
                 returns_dict=False):
        self.pp_dir = pp_dir
        self.subset = subset
        if isinstance(mask_keys, list):
            self.mask_keys = mask_keys
        elif isinstance(mask_keys, str):
            self.mask_keys = [mask_keys]
        else:
            self.mask_keys = list(mask_keys)
        self.build_path()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.returns_dict = returns_dict

    def __len__(self):
        """
        :return: int, the number of batches per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def build_path(self):
        subset_iter = (self.subset,) if isinstance(self.subset, str) else self.subset
        self.images_paths = []
        for subset in subset_iter:
            base_path = os.path.join(self.pp_dir, subset)
            study_uids = sorted(os.listdir(base_path))
            for study_uid in study_uids:
                img_path = dict()
                all_pp_data_exist = True
                keys = ['pet_img', 'ct_img'] + self.mask_keys
                for key in keys:
                    img_path[key] = os.path.join(base_path, study_uid, key + '.npy')
                    if not os.path.exists(os.path.join(base_path, study_uid, key + '.npy')):
                        all_pp_data_exist = False
                        print('study_uid {} : no preprocessed data for {}'.format(study_uid, study_uid, key))
                if all_pp_data_exist:
                    self.images_paths.append(img_path)

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: int, position of the batch in the Sequence
        :return: tuple of numpy array, (X_batch, Y_batch) of shape (batch_size, ...)
        """

        # select indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # prepare the batch
        X_batch = []
        Y_batch = []
        for idx in indexes:
            img_path = self.images_paths[idx]

            # load input image
            pet_array = np.load(img_path['pet_img'])
            ct_array = np.load(img_path['ct_img'])

            # load segmentation
            masks = []
            for key in self.mask_keys:
                masks.append(np.load(img_path[key]))
            if len(masks) == 0:
                mask_array = masks[0]
            else:
                masks = np.array(masks)
                mask_array = np.mean(masks, axis=0)

            # concatenate PET and CT
            PET_CT_array = np.stack((pet_array, ct_array), axis=-1)  # channels last

            # add 1 channel to mask
            MASK_array = np.expand_dims(mask_array, axis=-1)  # channels last

            # add it to the batch
            X_batch.append(PET_CT_array)
            Y_batch.append(MASK_array)

        if self.returns_dict:
            study_uids = []
            for idx in indexes:
                img_path = self.images_paths[idx]
                directory = os.path.dirname(img_path['pet_img'])
                study_uid = os.path.split(directory)[1]
                study_uids.append(study_uid)
            return {"study_uid": study_uids, 'img': np.array(X_batch), 'seg': np.array(Y_batch)}
        else:
            return np.array(X_batch), np.array(Y_batch)


