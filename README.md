# Breast-dense-tissue-segmentation

This is the sorce code for breast density segmentation.
The breast density segmentation contains two parts: out breast segmentaiton and dense tissue segmentation.
This repository includes the code of training breast-segmentation model and dense-tissue-segmentation model seperately.

## U-Net architecture

The figure below shows a U-Net architecture implemented in this repository.

![unet](UNet.png)

## Data
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903

This dataset shares MRI imaging and other data for 922 patients with invasive breast cancer. Their prone position axial breast MRI images were acquired by 1.5T or 3T scanners. Following MRI sequences are shared: a non-fat saturated T1-weighted sequence, a fat-saturated gradient echo T1-weighted pre-contrast sequence, and mostly three to four post-contrast sequences. Experiment in our paper uses the fat-saturated gradient echo T1-weighted pre-contrast sequence.

## Results
![dense_1](Breast_MRI_187_pre_081.png)
![dense_2](Breast_MRI_231_pre_076.png)
![dense_3](Breast_MRI_503_pre_105.png)
![dense_4](Breast_MRI_629_pre_100.png)
