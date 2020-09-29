import tensorflow as tf
import numpy as np
import os


class DataGenerator_3D_from_numpy(tf.keras.utils.Sequence):

    def __init__(self,
                 pp_dir,
                 subset,
                 mask_keys,
                 batch_size=1,
                 shuffle=True):
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

            pet_array = np.load(img_path['pet_img'])
            ct_array = np.load(img_path['ct_img'])

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

        return np.array(X_batch), np.array(Y_batch)


class DataGenerator_3D_from_nifti(tf.keras.utils.Sequence):

    def __init__(self,
                 images_paths,
                 transforms,
                 batch_size=1,
                 shuffle=True):

        self.images_paths = images_paths
        self.transforms = transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

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
        for idx in indexes:
            img_dict = self.images_paths[idx]
            if self.transforms is not None:
                for transform in self.transforms:
                    img_dict = transform(img_dict)

            # add it to the batch
            X_batch.append(img_dict)

        return X_batch


