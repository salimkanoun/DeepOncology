
from transforms import * 
def get_transform(subset, modalities, mode, method, tval, target_size, target_spacing, target_direction, target_origin = None,  data_augmentation=True, from_pp=False, cache_pp=False, mask_survival = None):
    """[summary]

    Args:
        
        subset ([str]): [train, val or test]
        modalities ([tuple]): [('pet_img, ct_img') or ('pet_img')]
        mode ([str]): [binary or probs]
        method ([str]): [ if binary, choose between : relative, absolute or otsu
                            else : method = []  ]
        tval ([float]) : [if mode = binary & method = relative : 0.41
                          if mode = binary & method = absolute : 2.5, 
                          else : don't need tval, tval = 0.0 ]

    """
    if mask_survival==True:
        keys = tuple(list(modalities) + ['mask_img'])
        dtypes = dtypes = {'pet_img': sitk.sitkFloat32,
                    'ct_img': sitk.sitkFloat32,
                    'mask_img': sitk.sitkVectorUInt8}
    else: 
        mask = False
        keys = tuple(list(modalities))
        dtypes = {'pet_img': sitk.sitkFloat32,
                        'ct_img': sitk.sitkFloat32}
    transformers = [LoadNifti(keys=keys)]  # Load NIFTI file from path

    transformers.append(ResampleReshapeAlign(target_size, target_spacing, target_direction, target_origin=None, keys=keys, test = mask))


    # Add Data augmentation
    
    if subset == 'train' and data_augmentation:
        translation = 10
        scaling = 0.1
        rotation = (np.pi / 60, np.pi / 30, np.pi / 60)
        transformers.append(RandAffine(keys=keys, translation=translation, scaling=scaling, rotation=rotation))
       


    # Convert Simple ITK image into numpy 3d-array
    transformers.append(Sitk2Numpy(keys=keys))
    # Normalize input values
    
    for modality in modalities:
        if modality == 'pet_img' : 
            modal_pp = dict(a_min=0.0, a_max=25.0, b_min=0.0, b_max=1.0, clip=True)
        else : 
            modal_pp = dict(a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True)

        transformers.append(ScaleIntensityRanged(keys = modality,
                                                 a_min =modal_pp['a_min'], a_max = modal_pp['a_max'], b_min=modal_pp['b_min'], b_max=modal_pp['b_max'], clip=modal_pp['clip']))

    # Concatenate modalities if necessary
    if mask_survival == True:
        transformers.append(ConcatModality(keys=keys, channel_first=False, new_key='input'))
    elif mask_survival == False:
        transformers.append(ConcatModality(keys=modalities, channel_first=False, new_key='input'))
    else:
        if len(modalities) > 1:
            transformers.append(ConcatModality(keys=modalities, channel_first=False, new_key='input'))
        else:
            transformers.append(AddChannel(keys=modalities, channel_first=False))
            transformers.append(RenameDict(keys=modalities, keys2='input'))

        transformers.append(AddChannel(keys='mask_img', channel_first=False))
    transformers = Compose(transformers)
    return transformers

