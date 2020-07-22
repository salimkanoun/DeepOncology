# MedicalSegmentation

Continuation of work of [Thomas Project](https://github.com/ThomasT3563/medical-segmentation) realised during a 6 months internship at IUCT Oncopole, France.

This provides some deep Learning tools for automatic segmentation of medical images. The approach implemented for this project is to process the whole body acquisition. One of the major challenges when processing this kind of data using deep learning algorithms is the memory usage, as depending on the modality and the study, an imaging serie can contains several hundreds or thousands of images.

### Model
The model used during this project is a custom [U-Net](https://arxiv.org/abs/1505.04597). The model [V-Net](https://arxiv.org/abs/1606.04797) is also implemented.

Implemented model :

- [x] 3D U-Net
- [x] V-Net
- [ ] Other model


### Segmentation of tumour on PET/CT scan

Trained model available: ```/master/deeplearning_models/trained_model_09241142.h5```

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="./GIF_example_segmentation.gif">
</p>


### Train model
- To train V-net, run :
`python3 training_cnn_lymphoma_segmentation.py` 


