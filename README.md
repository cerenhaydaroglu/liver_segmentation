# U-Net Segmentation for Medical Images by Pytorch

This repository provides an **extremely simple code** <img src="./fig/guaranteed.jpg" width="150"> for Image Segmentation by [U-Net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) using PyTorch. [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094) dataset is used for demonstration.

---

## Overview

### Data

Use open medical dataset [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094) with no preprocessing required.


### Data Structure
The original dataset contains **131 train** & **70 test** 3D CT images in **.nii** format. The 3D image sizes are (512, 512, 74\~987), where slice number varies.


### Model

![fig/unet.png](fig/unet.png)

The U-Net is implemented by Pytorch for deep learning. It is easy to visualize the architechture and convenient for changing your own dataset.

The inputs are grayscale CT images and output has 3 different classes in LiTS ("1": liver, "2": tumor, "0":anthing else)

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.


---

## How to use

### Dependencies

This example code runs with the following libraries:

* Python 3.6
* PyTorch >= 1.0.1
* [NiBabel (to read *.nii* files)](https://nipy.org/nibabel/)
* Matplotlib, Numpy, Scikit-Learn, Scikit-Image

### Computational device:
* at least 32Gb CPU memory
* at least 1 NVidia GPU with 11Gb (GPU) memory








### Run main.py

You will see the predicted results of test image in data/membrane/test




### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)

