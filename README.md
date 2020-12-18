# DeepOncology

This provides some deep Learning tools for automatic segmentation of 3D medical images (PET & CT scan).

### Models
Models used during this project are deep learning model like [U-Net](https://arxiv.org/abs/1505.04597). 

Implemented model :

- [x] [3D U-Net](https://arxiv.org/abs/1606.06650)
- [x] [V-Net](https://arxiv.org/abs/1606.04797)
- [x] [DenseX-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8946601).


### Segmentation of tumour on PET/CT scan

**Results :**

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="./GIF_example_segmentation.gif">
</p>


##  Installation
Setup package in a virtual environment:
```
git clone https://github.com/rnoyelle/DeepOncology.git
cd DeepOncology
conda create --name <env_name> python=3.7
source activate <env_name>
pip install -r requirements.txt
```

## Train new model
- To train V-Net :
```
source activate <env_name>
python training_3d_cnn.py --config config/config_3d.py
```

Then run this command to evaluate the performance
```
python evaluate_3d_cnn.py --config config/config_3d.py --weight path/to/weight.h5 -t result
```
And Explore result in the Jupyter Notebook : result_stats.ipynb



