# Modified by Alexander Shieh 2021/11/16, Yu-Tong Cheng 2022/06/02
# Code for testing: original model (global) + small model 1 (local: [0.2*n_slices:0.5*n_slices, 128:384, 128:384]) 
#                                           + small model 2 (local: [:0.2*n_slices, :, :])


#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5,7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load model name
GLOBAL_MODEL_NAME = "unet_dense_30_v3_suf_8690.hdf5"
LOCAL_MODEL_NAME_1 = "unet_dense_30_v3_suf_local_new_best.hdf5"
LOCAL_MODEL_NAME_2 = "unet_dense_30_v3_suf_local-0-02_best.hdf5"
mode = {4: 'test'} # 3: 'val'
data_path = "../../../tmp/pelvic_dataset_v2"

import glob
import nibabel as nib
import re
import csv
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import draw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Internal libraries/scripts
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

# self-defined libraries
from my_dicom_interface import My_DICOM_interface
import boundary_loss as bd
from mycc3d import ccResult

def combine(pred_glob, pred_local_1, pred_local_2):
    # start = timer()
    pred_glob[int(pred.shape[0]*0.2):int(pred.shape[0]*0.5), 128:384, 128:384] = np.logical_or(
        pred_glob[int(pred_glob.shape[0]*0.2):int(pred_glob.shape[0]*0.5), 128:384, 128:384], pred_local_1
    )
    """
    pred_glob[int(pred.shape[0]*0):int(pred.shape[0]*0.2), :, :] = np.logical_or(
        pred_glob[int(pred_glob.shape[0]*0):int(pred_glob.shape[0]*0.2), :, :], pred_local_2
    )
    """
    pred_glob[int(pred.shape[0]*0):int(pred.shape[0]*0.2), :, :] = pred_local_2;
    # end = timer()
    # print(f"combining global and local takes {end-start} s.", flush=True)

structure_dict = {"vessels": 1}
interface = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=False)
interface_local_1 = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=True, local_range=[0.2, 0.5])
interface_local_2 = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=False, local_range=[0.0, 0.2])

data_io = Data_IO(interface, data_path)
data_io_local_1 = Data_IO(interface_local_1, data_path)
data_io_local_2 = Data_IO(interface_local_2, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
samples = interface.initialize(data_path)
print("All samples: " + str(sample_list))

sample_list_local_1 = data_io_local_1.get_indiceslist()
sample_list_local_1.sort()
samples_local_1 = interface_local_1.initialize(data_path)
print("All samples: " + str(sample_list_local_1))

sample_list_local_2 = data_io_local_2.get_indiceslist()
sample_list_local_2.sort()
samples_local_2 = interface_local_2.initialize(data_path)
print("All samples: " + str(sample_list_local_2))

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
pp_local_1 = Preprocessor(data_io_local_1, data_aug=data_aug, batch_size=1, subfunctions=subfunctions, prepare_subfunctions=True, 
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(128, 256, 256))
pp_local_2 = Preprocessor(data_io_local_2, data_aug=data_aug, batch_size=1, subfunctions=subfunctions, prepare_subfunctions=True, 
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(128, 256, 256))

# METRICS


# Define metrics
# ref: https://github.com/frankkramer-lab/MIScnn/issues/79
def dice_nb(y_true, y_pred, smooth=0.00001):
    y_true=y_true[:,:,:,:,1:]
    y_pred=y_pred[:,:,:,:,1:]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# NETWORK
# Create the Neural Network model
# model = Neural_Network(preprocessor=pp, architecture=unet_dense, loss=bd.tl_sl_wrapper(alpha), metrics=metric_list, batch_queue_size=2, workers=2, learninig_rate=0.0001)

model = Neural_Network(preprocessor=pp, architecture=unet_dense, loss=tversky_crossentropy, metrics=dice_nb, batch_queue_size=2, workers=2, learninig_rate=0.00005)
model_local_1 = Neural_Network(preprocessor=pp_local_1, architecture=unet_dense, loss=tversky_crossentropy, metrics=dice_nb, batch_queue_size=2, workers=2, learninig_rate=0.00005)
model_local_2 = Neural_Network(preprocessor=pp_local_2, architecture=unet_dense, loss=tversky_crossentropy, metrics=dice_nb, batch_queue_size=2, workers=2, learninig_rate=0.00005)

# model.reset_weights()

# start = timer()

model.load(GLOBAL_MODEL_NAME)
model_local_1.load(LOCAL_MODEL_NAME_1)
model_local_2.load(LOCAL_MODEL_NAME_2)

# end = timer()
# print(f"loading models takes {end-start} s.")

def simple_dice(truth, pred):
    try:
        pd = np.equal(pred, 1)
        gt = np.equal(truth, 1)
        dice = 2*np.logical_and(pd, gt).sum()/(pd.sum() + gt.sum())
        return dice
    except ZeroDivisionError:
        return 0.0

def simple_dice2(truth, pred, smooth=0.00001):
    y_true_f = truth.flatten()
    y_pred_f = pred.flatten()
    intersection = np.sum(truth * pred)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# POST-PROCESSING
def cc(sample, slices):
    result = ccResult(sample, slices)
    result.remove_part('not-max', percentage=0.15)
    result.remove_part('dust', percentage=1.0)
    result.set_to_one()
    return result.labels

# PLOTTING IMAGES
def plot3d(data, dir_name, filename):
    # print("plot3d"+filename,flush=True)  
    # start = timer()
    
    x1 = []; y1 = []; z1 = []
    for z in range(data.shape[0]):
        for x in range(data.shape[2]):
            for y in range(data.shape[1]):
                if data[z][x][y] == 1:
                    x1.append(x)
                    y1.append(y)
                    z1.append(z)
                if data[z][x][y] not in [0,1]:
                    print(f"data[{z}][{x}][{y}]={data[z][x][y]}")
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1,y1,z1,c='red',s=1e-3)
    ax.set_box_aspect((0.429688, 0.429688, 0.625))

    try:
        os.mkdir(dir_name)
    except:
        pass
    views = {"Front":(0,0)} #, "RAO_60":(0,-60), "LAO_60":(0,60), "Caudal_40":(40,0), "Cranial_55":(-55,0)}            
    for angle in views.keys():
        view = views[angle]
        ax.view_init(view[0],view[1])
        plt.savefig(dir_name+'/'+angle+'_'+filename,dpi=300)
        print(dir_name+'/'+angle+'_'+filename, flush=True)
        
    # end = timer()
    # print(f"spent {end-start} s.",flush=True)

##

## TESTING ##
for mod in mode.keys(): # 'val', 'test'
    print(mode[mod], flush=True)
    mode_dir_name = mode[mod]
    try:
        os.mkdir(mode_dir_name)
    except:
        pass

    dice_list = []

    test_list = sample_list[mod::5]
    test_list_local_1 = sample_list_local_1[mod::5]
    test_list_local_2 = sample_list_local_2[mod::5]
    test_list_num  = [mod+k*5 for k in range(10)]
    # print(test_list_num)

    # start = timer()
    predictions = model.predict(test_list, return_output=True, activation_output=False)
    predictions_local_1 = model_local_1.predict(test_list_local_1, return_output=True, activation_output=False)
    predictions_local_2 = model_local_2.predict(test_list_local_2, return_output=True, activation_output=False)
    # end = timer()
    # print(f"making {len(test_list)} predictions takes {end-start} s.",flush=True)


    dice_sum_global = 0.0
    dice_sum_local_1 = 0.0
    dice_sum_local_2 = 0.0
    dice_sum_combined = 0.0
    dice_sum_cc_after_combined = 0.0

    truth_flat_global = np.array([])
    pred_flat_global = np.array([])
    truth_flat_local_1 = np.array([])
    pred_flat_local_1 = np.array([])
    truth_flat_local_2 = np.array([])
    pred_flat_local_2 = np.array([])
    pred_flat_combined = np.array([])
    pred_flat_cc_after_combined = np.array([])

    # for each data
    for k in range(len(test_list)):
        ID = test_list_num[k]
        print(f"CVAI{ID+1}\t", flush=True)
        print(predictions[k][0])

        # start = timer()
        tmp_shape = predictions[k].shape
        sample = data_io.sample_loader(sample_list[ID])
        sample_local_1 = data_io_local_1.sample_loader(sample_list_local_1[ID])
        sample_local_2 = data_io_local_2.sample_loader(sample_list_local_2[ID])
        # end = timer()
        # print(f"loading data{k} takes {end-start} s.",flush=True)

        # ground truth: global
        plot3d(sample.seg_data, mode_dir_name+'/gt_g', 'gt_g_'+str(ID+1)+'.png')
        truth_flat_global = np.concatenate((truth_flat_global, sample.seg_data), axis=None)

        # ground truth: local
        plot3d(sample_local_1.seg_data, mode_dir_name+'/gt_l1', 'gt_l1_'+str(ID+1)+'.png')
        truth_flat_local_1 = np.concatenate((truth_flat_local_1, sample_local_1.seg_data), axis=None)
        plot3d(sample_local_2.seg_data, mode_dir_name+'/gt_l2', 'gt_l2_'+str(ID+1)+'.png')
        truth_flat_local_2 = np.concatenate((truth_flat_local_2, sample_local_2.seg_data), axis=None)

        # prediction: global
        pred_flat_global = np.concatenate((pred_flat_global, predictions[k][:,:,:, np.newaxis]), axis=None)
        plot3d(predictions[k], mode_dir_name+'/pr_g', 'pr_g_'+str(ID+1)+'.png')

        dice_global = simple_dice2(sample.seg_data, predictions[k][:,:,:, np.newaxis])
        dice_sum_global += dice_global

        # prediction: local
        pred_flat_local_1 = np.concatenate((pred_flat_local_1, predictions_local_1[k][:,:,:, np.newaxis]), axis=None)   
        plot3d(predictions_local_1[k], mode_dir_name+'/pr_l1', 'pr_l1_'+str(ID+1)+'.png')
        pred_flat_local_2 = np.concatenate((pred_flat_local_2, predictions_local_2[k][:,:,:, np.newaxis]), axis=None)   
        plot3d(predictions_local_2[k], mode_dir_name+'/pr_l2', 'pr_l2_'+str(ID+1)+'.png')

        dice_local_1 = simple_dice2(sample_local_1.seg_data, predictions_local_1[k][:,:,:, np.newaxis])
        dice_sum_local_1 += dice_local_1
        dice_local_2 = simple_dice2(sample_local_2.seg_data, predictions_local_2[k][:,:,:, np.newaxis])
        dice_sum_local_2 += dice_local_2

        pred = predictions[k] # global
        # prediction: combine
        combine(pred, predictions_local_1[k], predictions_local_2[k])
        pred_flat_combined = np.concatenate((pred_flat_combined, predictions[k][:,:,:, np.newaxis]), axis=None)
        plot3d(pred, mode_dir_name+'/pr_cb-no-cc', 'pr_cb-no-cc_'+str(ID+1)+'.png')

        dice_combined = simple_dice2(sample.seg_data, pred[:,:,:, np.newaxis])
        dice_sum_combined += dice_combined

        # prediction: cc3d (after combine)
        pred = cc(pred, tmp_shape[0])
        pred_flat_cc_after_combined = np.concatenate((pred_flat_cc_after_combined, pred[:,:,:, np.newaxis]), axis=None)
        plot3d(pred, f'{mode_dir_name}/pr_cc-after-cb', f'pr_cc-after-cb_{str(ID+1)}.png')

        dice_cc_after_combined = simple_dice2(sample.seg_data, pred[:,:,:, np.newaxis])
        dice_sum_cc_after_combined += dice_cc_after_combined

        # print dice scores
        dice_list.append(["CVAI"+str(ID+1), dice_global, dice_local_1, dice_local_2, 
            dice_combined, dice_cc_after_combined])
        print(dice_list[-1])

    # end of for each data

    print("Dice scores:")

    print("\n # \t\t dice_global \t dice_local_1 \t dice_local_2 \t dice_combined", end='')
    print(f"\t dice_cc_after_combined")
    for dice in dice_list:
        print(dice)
    print(f"Average: \t {dice_sum_global/len(test_list)} ", end='')
    print(f"\t {dice_sum_local_1/len(test_list)} \t {dice_sum_local_2/len(test_list)}",end='')
    print(f"\t {dice_sum_combined/len(test_list)} \t {dice_sum_cc_after_combined/len(test_list)}")
    print(f"Flattened: \t {simple_dice2(truth_flat_global, pred_flat_global)}", end="");
    print(f"\t {simple_dice2(truth_flat_local_1, pred_flat_local_1)}", end='')
    print(f"\t {simple_dice2(truth_flat_local_2, pred_flat_local_2)}", end='')
    print(f"\t {simple_dice2(truth_flat_global, pred_flat_combined)}", end='')
    print(f"\t {simple_dice2(truth_flat_global, pred_flat_cc_after_combined)}")
    print(flush=True)

# end of for each mode
