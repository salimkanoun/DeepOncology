# MedicalSegmentation

Continuation of  [Thomas Project](https://github.com/ThomasT3563/medical-segmentation) realised during a 6 months internship at IUCT Oncopole, France.

This provides some deep Learning tools for automatic segmentation of medical images. The approach implemented for this project is to process the whole body acquisition. One of the major challenges when processing this kind of data using deep learning algorithms is the memory usage, as depending on the modality and the study, an imaging serie can contains several hundreds or thousands of images.

### Model
The model used during this project is [U-Net](https://arxiv.org/abs/1505.04597). 
Other models are also implemented like [V-Net](https://arxiv.org/abs/1606.04797) or [DenseX-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8946601).

Implemented model :

- [x] 3D U-Net
- [x] V-Net
- [ ] DenseX-Net


### Segmentation of tumour on PET/CT scan

**Results :**

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="./GIF_example_segmentation.gif">
</p>


### Train new model
- To train V-net :
> `python3 training_cnn_lymphoma_segmentation.py config/default_config.json` 

### Check data
- To generate 2D MIP :
> `python3 generate_pdf.py`

- To transform raw data to preprocessed numpy array :
> `python3 prep_data.py`



