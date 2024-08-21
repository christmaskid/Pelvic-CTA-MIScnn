# Modified by Alexander Shieh 2021/11/16
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import glob
import nibabel as nib
import re
import csv
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import draw

# Internal libraries/scripts
from my_dicom_interface import My_DICOM_interface

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

#from miscnn.neural_network.architecture.unet.attention import Architecture
#unet_attention = Architecture()

from miscnn.utils.visualizer import visualize_evaluation

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Create Interface
# interface = My_DICOM_interface(annotation_tag="vessels")

# Initialize 
data_path = "../../../tmp/pelvic_dataset_v2"
# samples = interface.initialize(data_path)

# Obtain ROI names
# structures = interface.get_ROI_names(samples[0])
# print('Found structures in sample : {}'.format(structures))

structure_dict = {"vessels": 1}
interface = My_DICOM_interface(structure_dict = structure_dict, classes=2, annotation_tag="vessels", local=False)
#interface_local = My_DICOM_interface(structure_dict = structure_dict, classes=2, annotation_tag="vessels", local=True)

data_io = Data_IO(interface, data_path)
#data_io_local = Data_IO(interface_local, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
samples = interface.initialize(data_path)
print("All samples: " + str(sample_list))

#sample_list_local = data_io_local.get_indiceslist()
#sample_list_local.sort()
#samples_local = interface_local.initialize(data_path)
#print("All samples: " + str(sample_list_local))

# DATASET
# Create train, validation, and test list
train_list = sample_list[0::5]+sample_list[1::5]+sample_list[2::5]
val_list = sample_list[3::5]
test_list = sample_list[4::5]

#train_list_local = sample_list_local[0::5]+sample_list[1::5]+sample_list_local[2::5]
#val_list_local = sample_list_local[3::5]
#test_list_local = sample_list_local[4::5]

print(f"train_list: {[0+5*i for i in range(10)]+[1+5*i for i in range(10)]+[2+5*i for i in range(10)]}")
print(f"val_list: {[3+5*i for i in range(10)]}")
print(f"test_list: {[4+5*i for i in range(10)]}")

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)


# PREPROCESSING
# Create a pixel value normalization Subfunction through Z-Score 
sf_normalize = Normalization(mode="z-score")
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((0.625, 0.429688, 0.429688))
# Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
sf_clipping = Clipping(min=-200, max=800)
# Assemble Subfunction classes into a list
# Be aware that the Subfunctions will be exectued according to the list order!
subfunctions = [sf_clipping, sf_normalize, sf_resample]
# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=1, subfunctions=subfunctions, prepare_subfunctions=True, 
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(128, 256, 256))


# METRICS
import my_losses
import matplotlib.pyplot as plt

# new_loss_0 = combo_loss()
new_loss_1 = symmetric_focal_loss()
new_loss_2 = symmetric_focal_tversky_loss()
new_loss_3 = sym_unified_focal_loss()

# Define metrics
# ref: https://github.com/frankkramer-lab/MIScnn/issues/79
def dice_nb(y_true, y_pred, smooth=0.00001):
    y_true=y_true[:,:,:,:,1:]
    y_pred=y_pred[:,:,:,:,1:]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

metric_list = [dice_nb, dice_soft, dice_coefficient]


# CALLBACK
# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=1e-14)
#cb_lr_local = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=1e-10)
# cb_es = EarlyStopping(monitor='loss', min_delta=0.01, patience=5, verbose=1, mode='min')
cb_cp = ModelCheckpoint("unet_dense_30_v3_suf_best.hdf5", monitor='val_dice_nb', mode='max', save_best_only=True, verbose=1, perioid=1)
#cb_cp_local = ModelCheckpoint("unet_dense_30_v3_uf_local_best.hdf5", monitor='val_dice_nb', mode='max', save_best_only=True, verbose=1, perioid=1)
# cb_bd = bd.AlphaScheduler(alpha, bd.update_alpha)


# NETWORK
# Create the Neural Network model

model = Neural_Network(preprocessor=pp, architecture=unet_dense, loss=new_loss_3, metrics=metric_list, batch_queue_size=2, workers=2, learninig_rate=1e-4)
history = model.evaluate(training_samples=train_list,
                         validation_samples=val_list,
                         epochs=60,
                         iterations=None,
                         callbacks=[cb_lr, cb_cp]
                         )
model.dump("unet_dense_30_v3_suf.hdf5")
#print(history.history.keys())
"""
train_loss = history.history['loss']
val_loss = history.history['val_loss']
xc = range(1,61)
plt.figure()
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.title("Global model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','val'])
plt.savefig("global_loss.png")
"""

"""
model_local = Neural_Network(preprocessor=pp, architecture=unet_attention, loss=new_loss_3, metrics=metric_list, batch_queue_size=2, workers=2, learninig_rate=0.00005)
model_local.load("unet_dense_30_v3_uf_best.hdf5")
history = model_local.evaluate(training_samples=train_list_local,
                         validation_samples=val_list_local,
                         epochs=30,
                         iterations=None,
                         callbacks=[cb_lr_local, cb_cp_local]
                         )
model_local.dump("unet_dense_30_v3_uf_local.hdf5")
train_loss = history.history['loss']
val_loss = history.history['val_loss']
xc = range(1,31)
plt.figure()
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.title("Local model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','val'])
plt.savefig("local_loss.png")
"""
