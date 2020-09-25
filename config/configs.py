
# from default_configs import DefaultConfigs
# class configs(DefaultConfigs):


class configs(object):

    def __init__(self):

        # path
        self.csv_path = "/home/salim/Documents/DeepOncopole/data/DB_PATH_NIFTI.csv"
        self.trained_model_path = None
        self.training_model_folder = "/media/salim/DD 2To/RUDY_WEIGTH/training"

        # preprocessing
        self.image_shape = (256, 128, 128)  # (z, y, x)
        self.voxel_spacing = (4.0, 4.8, 4.8)  # (z, y, x)
        self.in_channels = 2  # PET + CT = 2
        self.out_channels = 1  # sigmoid
        self.number_class = 2  # Background + lymphoma

        self.data_augment = True
        self.origin = "head"
        self.normalize = True

        self.threshold = 'otsu'

        # model
        self.model = 'vnet'
        {'vnet': self.add_vnet_config}[self.model]

        # training
        self.epochs = 100
        self.batch_size = 2
        self.schuffle = True
        self.opt = "SGD"
        self.opt_params = {
                "learning_rate": 1e-03,
                "momentum": 0.9
            }

        # callbacks
        self.ReduceLROnPlateau = False
        self.EarlyStopping = False
        self.patience = 10
        self.ModelCheckpoint = True
        self.TensorBoard = True

    def add_vnet_config(self):
        self.kernel_size = (5, 5, 5)
        self.number_channels = 8
        self.num_levels = 4
        self.num_convolutions = (1, 2, 3, 3)
        self.bottom_convolutions = 3
        self.activation = "relu"

        self.keep_prob = 1.0  # 1.0 =  no dropout
