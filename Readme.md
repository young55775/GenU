from numpy.f2py.symbolic import as_numer_denom

# GenU: Cytoskeleton and Vesicle segmentation with generated dataset and U-net

### requirement
- Pytorch==2.5.1
- Scikit-learn==1.6.0
- Scipy==1.14.1
- opencv-python==4.10.0.84
- numpy==2.2.0
Recommend to install all the packages by 
```commandline
#bash
pip install ultralytics
```
Ultralytics is a well documented yolo package by install which you can get almost everything you need for CV deep learning tasks.

### Main
GenU is a U-net trained with generated datasets, It is a novel pipeline that generates biologically inspired synthetic 
datasets and trains a U-net architecture for subcellular structure segmentation. Our method eliminates reliance on 
experimental or physics-based simulation data, drastically simplifying the dataset creation process while preserving 
biological relevance. Using 1,000 synthetic images, we trained a basic U-Net for only five epochs, achieving good 
performance in segmenting structures such as microtubules and vesicles—without any fine-tuning on any microscopy 
photograph data. 
![img_3.png](img_3.png)

```requirements
Vesicle_generator.py
Cytoskeleton_generator.py
```
are scripts used to generate training and validation dataset shown above.

```requirements
GenU_cytoskeleton.py
GenU_Vesicle.py
```
are scripts used for cytoskeleton and vesicle segmentation

Tiff files are needed for both script.
Vesicle segmentation needs 16-bit tiff pictures with single channel.
Cytoskeleton segmentation needs 8-bit tiff pictures with single channel.
Both will return a binary mask, of which the threshold can be adjusted in the scripts.
The results of segmentation were shown below:


![img_4.png](img_4.png)
![img_2.png](img_2.png)
![img_1.png](img_1.png)
