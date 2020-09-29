import argparse
import json


def main(config):





    # Get Data
    if pp_dir is None:
        DM = DataManager(csv_path=csv_path)
        x_train, x_val, x_test, y_train, y_val, y_test = DM.get_train_val_test()

        # Define generator
        train_generator = DataGenerator(x_train, y_train,
                                        batch_size=batch_size, shuffle=shuffle, augmentation=data_augment,
                                        target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                        resize=resize, normalize=normalize, origin=origin)

        val_generator = DataGenerator(x_val, y_val,
                                      batch_size=batch_size, shuffle=False, augmentation=False,
                                      target_shape=image_shape, target_voxel_spacing=voxel_spacing,
                                      resize=resize, normalize=normalize, origin=origin)
    else:
        mask_keys = ['mask_img_absolute', 'mask_img_relative', 'mask_img_otsu']
        train_generator = DataGenerator_3D_from_numpy(pp_dir, 'train',
                                                      mask_keys,
                                                      batch_size=batch_size,
                                                      shuffle=True)
        val_generator = DataGenerator_3D_from_numpy(pp_dir, 'val',
                                                    mask_keys,
                                                    batch_size=batch_size,
                                                    shuffle=False)

        test_generator = DataGenerator_3D_from_numpy(pp_dir, 'test',
                                                    mask_keys,
                                                    batch_size=batch_size,
                                                    shuffle=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='config/default_config.json', type=str,
                        help="path/to/config.json")
    parser.add_argument("--model_path", type=str,
                        help="path/to/config.json")
    parser.add_argument("--target_dir", type=str,
                        help="path/to/config.json")
    args = parser.parse_args()

    if args.model_path is None:
        raise argparse.ArgumentError(parser,
                                     "model path was noy supplied.\n" + parser.format_help())

    with open(args.config) as f:
        config = json.load(f)

    main(config, args)


