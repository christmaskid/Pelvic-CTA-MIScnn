# Pelvic-CTA-MIScnn

Code for training model on pelvic CTA images using **dense U-Net** architecture based on package **MIScnn** (https://github.com/frankkramer-lab/MIScnn/) .

## Generating training data

## Data I/O

Defined in [my_dicom_interface.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/my_dicom_interface.py).

Revised from https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/data_loading/interfaces/dicom_io.py.

## Loss Functions

Defined in [my_losses.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/my_losses.py).

References: 
* Most loss functions: https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/neural_network/metrics.py
* Surface loss: https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py

## Model Training
See [train_MIScnn_dense_global.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/train_MIScnn_dense_global.py).

[train_MIScnn_dense_local.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/train_MIScnn_dense_local.py) trains the second model transferred from the first one. 

The second model focuses on a small part of the original sample, i.e., the middle 256 * 256 * 'some range of slices' voxels, in order to perform better on the internal pudendal arteries (IPA).

* Input: 
  * images: Image dicom files.
  * labels: Ground truth in csv file format.
* Output: 3d numpy array.

## Testing
See [test_MIScnn_dense_2model.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/test_MIScnn_dense_2model.py)

Testing is done after post-processing by [mycc3d.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/postprocessing/mycc3d_scipy.pyx) according to the diagram below.
* Package used: https://github.com/seung-lab/connected-components-3d.

![fig-1](https://user-images.githubusercontent.com/66014047/218907938-331bc0c3-29c9-49ba-80dd-e763c628e3f8.png)


(If only one or neither of the local models are trained, delete the part for the not-trained models.)

* Input: images & labels
* Output: Prediction results (3d numpy array), 3d plots and dice scores.

## Centerline Extraction
Centerline extraction is done by skeletonization, using package scikit-image (```skimage.morphology.skeletonize3d```). 

Results are saved as binary map, visualized by HTML files generated by package ```plotly```.

See [test_MIScnn_dense_2model_centerline.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/codes/postprocessing/test_MIScnn_dense_2model_centerline.py)

## Experiments and Results
  
| Model      | Global |        | Local  | Combined |        | Post-processed |        |
|:----------:|:------:|:------:|:------:|:--------:|:------:|:--------------:|:------:|
| Range  | global | local | local  | global | local  | global   | local  |
| Dice score | 0.8153| 0.8282 | 0.8249 | 0.8825 | 0.8483 | 0.8818 | 0.8788 |

