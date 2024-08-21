# A model training script based on miscnn package.
# Global
# Modified by Alexander Shieh 2022/03/03, Yu-Tong Cheng 2022/07/04

#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # set visible GPUs

import glob
import nibabel as nib
import re
import csv
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import draw

# Internal scripts
from my_dicom_interface import My_DICOM_interface
import my_losses

# Library import
from miscnn.data_loading.interfaces.dicom_io import DICOM_interface
from miscnn.data_loading.data_io import Data_IO
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn import Preprocessor

from miscnn import Neural_Network
from miscnn.neural_network.metrics import *
# from miscnn.neural_network.architecture.unet.standard import Architecture
# unet_standard = Architecture()

from miscnn.neural_network.architecture.unet.dense import Architecture
unet_dense = Architecture()

from miscnn.utils.visualizer import visualize_evaluation

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Create Interface
structure_dict = {"vessels": 1}
interface = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=False, local_range=[0, 1.0])
# local: get 256*256 voxels in the middle from x-y plane
# local_range: range of slices

# Initialize 
data_path = "../../../tmp/pelvic_dataset_v2"
samples = interface.initialize(data_path)
sample = data_io.sample_loader(samples[0])
images, segmentations = sample.img_data, sample.seg_data
print(images.shape, segmentations.shape)

# Obtain ROI names
# structures = interface.get_ROI_names(samples[0])
# print('Found structures in sample : {}'.format(structures))

data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
print("All samples: " + str(sample_list))


# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)


# PREPROCESSING
# Create a pixel value normalization Subfunction through Z-Score 
sf_normalize = Normalization(mode="z-score")
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((0.625, 0.429688, 0.429688))
# Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
sf_clipping = Clipping(min=-200, max=1500)
# Assemble Subfunction classes into a list
# Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_clipping, sf_normalize, sf_resample]
# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=1, subfunctions=subfunctions, prepare_subfunctions=True, 
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(128, 256, 256))


# METRICS

new_loss_0 = combo_loss()
new_loss_1 = asymmetric_focal_loss()
new_loss_2 = asymmetric_focal_tversky_loss()
new_loss_3 = asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5)

# ref: https://github.com/frankkramer-lab/MIScnn/issues/79
# Dice score without background has not been implemented yet in miscnn package.
def dice_nb(y_true, y_pred, smooth=0.00001):
    y_true=y_true[:,:,:,:,1:]
    y_pred=y_pred[:,:,:,:,1:]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

metric_list = [dice_nb, dice_soft, dice_coefficient, new_loss_3, tversky_crossentropy]

# CALLBACK
# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=1e-12)
# cb_es = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='min')
cb_cp = ModelCheckpoint("unet_dense_30_v3_suf_local-0-02_best.hdf5", monitor='val_dice_nb', mode='max', save_best_only=True, verbose=1, perioid=1)

# Scheduler for surface loss (see my_losses.surface_loss_keras and tl_sl_wrapper)
alpha = K.variable(1, dtype='float32')
cb_bd = AlphaScheduler(alpha, update_alpha)

# NETWORK
# Create the Neural Network model
# model = Neural_Network(preprocessor=pp, architecture=unet_dense, loss=tl_sl_wrapper(alpha), metrics=metric_list, batch_queue_size=2, workers=2, learninig_rate=0.0001)
### 'learninig_rate' is mis-spelled in miscnn. That's not a typo. ###

model = Neural_Network(preprocessor=pp, architecture=unet_dense, loss=new_loss_3, metrics=metric_list, batch_queue_size=2, workers=2, learninig_rate=0.00001)
# model.reset_weights()

test_list = sample_list[4::5]
val_list = sample_list[3::5]
train_list = sample_list[0::5] + sample_list[1::5] + sample_list[2::5]
print(test_list, val_list, train_list, flush=True)

history = model.evaluate(training_samples=train_list,
                         validation_samples=val_list,
                         epochs=30,
                         iterations=None,
                         callbacks=[cb_lr, cb_cp, cb_bd]
                         )

# model.train(sample_list[0:2], epochs=1)

predictions = model.predict(sample_list[27:], return_output=True)
# sample = data_io.sample_loader(sample_list[2], load_seg=True, load_pred=True)
# img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
sample = data_io.sample_loader(sample_list[29])

# Below can only be used in miscnn <= 1.3.0
visualize_evaluation(sample_list[29], sample.img_data, sample.seg_data, predictions[2][:,:,:, np.newaxis], "plot_directory/")

