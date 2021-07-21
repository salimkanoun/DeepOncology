
import numpy as np 

class DataGeneratorSurvival(tf.keras.utils.Sequence):
    """A class to create a DataGenerator object for model.fit()"""

    def __init__(self,
                 images_paths, 
                 struct_data,
                 y_train,
                 transforms,
                 x_key,
                 batch_size,
                 shuffle=True):
        """
        :param images_paths: list[dict] or dict[dict]
        :param transforms: transformer to apply to data
        :param batch_size: batch size
        :param shuffle: bool. If set to true, indexes will be suffled at each end of epoch.
        :param x_key: key corresponding to input of neural network

        """

        self.images_paths = images_paths
        self.struct_data = struct_data
        self.y_train = y_train
        self.transforms = transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_key = x_key
        self.on_epoch_end()

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
        X_batch_struct_data = []
        Y_batch = [[]]
        for idx in indexes:
            img_dict = self.images_paths[idx]
            if self.transforms is not None:
                img_dict = self.transforms(img_dict)
            # add it to the batch
            X_batch.append(img_dict[self.x_key])
            Y_batch[0].append(self.y_train[idx])
            #Y_batch_test[0].append(self.y_train[idx][0])
            #Y_batch_test[1].append(self.y_train[idx][1])
            if len(self.struct_data)!=0:
                X_batch_struct_data.append(self.struct_data[idx])
        X_batch = np.array(X_batch)
        
        Y_batch = np.asarray(Y_batch).astype('int32')
        if len(self.struct_data)!=0: 
            X_batch_struct_data = np.array(X_batch_struct_data)
            X_batch= [X_batch_struct_data, X_batch]
        return X_batch, Y_batch

