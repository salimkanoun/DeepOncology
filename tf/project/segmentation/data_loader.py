import tensorflow as tf
import numpy as np

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
        #print('PREPARE THE BATCH')
        #start_time = time.time()
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
        #print("END BATCH --- %s seconds ---" % (time.time() - start_time))
        return X_batch, Y_batch



