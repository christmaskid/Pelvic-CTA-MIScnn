# Pelvic-CTA-MIScnn

Code for training model on pelvic CTA images using dense U-Net architecture based on package MIScnn (https://github.com/frankkramer-lab/MIScnn/) .

## Data I/O

Defined in [my_dicom_interface.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/my_dicom_interface.py).

Revised from https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/data_loading/interfaces/dicom_io.py.

## Loss Functions

Defined in [my_losses.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/my_losses.py).

References: 
* Most loss functions: https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/neural_network/metrics.py
* Surface loss: https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py

## Model Training
Refer to [train_MIScnn_dense.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/train_MIScnn_dense.py).

## Testing

Testing is done after post-processing by [mycc3d.py](https://github.com/christmaskid/Pelvic-CTA-MIScnn/blob/main/mycc3d.py) according to the diagram below.

Package used: https://github.com/seung-lab/connected-components-3d.

(code: TO BE UPDATED)

## Experiments and Results
(TO BE UPDATED)

## Getting started
(TO-DO)
