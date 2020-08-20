# MedicalSegmentation

Continuation of  [Thomas Project](https://github.com/ThomasT3563/medical-segmentation) realised during a 6 months internship at IUCT Oncopole, France.

This provides some deep Learning tools for automatic segmentation of medical images (PET & CT scan).

### Model
The model used during this project are deep learning model like [U-Net](https://arxiv.org/abs/1505.04597). 

Implemented model :

- [x] [3D U-Net](https://arxiv.org/abs/1606.06650)
- [x] [V-Net](https://arxiv.org/abs/1606.04797)
- [x] [DenseX-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8946601).


### Segmentation of tumour on PET/CT scan

**Results :**

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="./GIF_example_segmentation.gif">
</p>


### Train new model
- To train V-Net :
> `python3 training_3d_cnn.py config/default_config.json` 
- To train DenseX-Net :
> `python3 training_2d_cnn.py config/default_config_2d.json` 

### Check data
- To generate 2D MIP :
> `python3 generate_pdf.py`

- To transform raw data to preprocessed numpy array :
> `python3 prep_data.py`



