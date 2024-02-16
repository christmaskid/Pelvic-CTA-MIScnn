# Pelvic-CTA-MIScnn

Code for training model on pelvic CTA images using **dense U-Net** architecture based on package **MIScnn** (https://github.com/frankkramer-lab/MIScnn/) .

More info and current progress: see https://drive.google.com/file/d/1M7qRDX4sSol-yzskj3qJp4TyOqgoI0-9/view?usp=sharing.

## Generating training data

### Pre-processing on labeling
Labels are the result of the manually adjusted ISP segmentation combined with the contours generated by ISP after adjustment of inner and outer vessel wall of IPA are performed in MPR coordination mode.
* Input: ISP dicom files
* Output: 3d coordinates (N * 3 arrays) or 3d masks.

See [my_mpr_to_mask_3d-interp.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/pre-processing/my_mpr_to_mask_3d-interp.py) for methods on transforming contours into solid segmentation.

### Converting files
Masks are generated from ISP dicom files using [read_segmentation_from_isp_ava.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/pre-processing/read_segmentation_from_isp_ava.py).
* Input: DICOM files from ISP.
* Output: 3d coordinates (N * 3 arrays) are saved into csv files. The data would be converted to 3d masks in the DATA I/O interface.

## Data I/O

Defined in [my_dicom_interface.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/train/my_dicom_interface.py).

Revised from https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/data_loading/interfaces/dicom_io.py.

## Loss Functions

Defined in [my_losses.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/train/my_losses.py).

References: 
* Most loss functions: https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/neural_network/metrics.py
* Surface loss: https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py

## Model Training
See [train_MIScnn_dense.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/train/train_MIScnn_dense.py).
[train_MIScnn_dense_local.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/train/train_MIScnn_dense_local.py) trains the second model transferred from the first one. The second model focuses on a small part of the original sample, i.e., the middle 256 * 256 * 'some range of slices' voxels, in order to perform better on the internal pudendal arteries (IPA).

* Input: 
  * images: Image dicom files.
  * labels: Ground truth in csv file format.
* Output: 3d numpy array.

## Testing
See [test_g+l+cb+cc_new_2local.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/test/test_g+l+cb+cc_new_2local.py)

Testing is done after post-processing by [mycc3d.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/test/mycc3d.py) according to the diagram below.
* Package used: https://github.com/seung-lab/connected-components-3d.

![fig-1](https://user-images.githubusercontent.com/66014047/218907938-331bc0c3-29c9-49ba-80dd-e763c628e3f8.png)


(If only one or neither of the local models are trained, delete the part for the not-trained models.)

* Input: images & labels
* Output: Prediction results (3d numpy array), 3d plots and dice scores.

## Experiments and Results



| Model      | Global | Local  | Combined | Post-processed |
|------------|--------|--------|----------|----------------|
| Dice score | 0.8443 | 0.7250 | 0.8464   | 0.8781         |


## Citation
Abstract No. 148 Detailed Segmentation of Pelvic Arteries in Pelvic CT Angiography with Deep Learning
A. Shieh, Y. Cheng, W. Lee, T. Wang. Journal of Vascular and Interventional Radiology, VOLUME 34, ISSUE 3, SUPPLEMENT , S69, MARCH 2023. DOI:https://doi.org/10.1016/j.jvir.2022.12.201
