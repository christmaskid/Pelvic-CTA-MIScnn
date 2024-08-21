# Modified by Alexander Shieh 2021/11/16
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load model name
GLOBAL_MODEL_NAME = "unet_dense_30_v3_suf_best.hdf5"
LOCAL_MODEL_NAME_1 = "unet_dense_30_v3_suf_local_best.hdf5"
mode = {4: 'test'} # 3: 'val'
data_path = "../../../tmp/pelvic_dataset_v2/"

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
import csv

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
from mycc3d import ccResult

structure_dict = {"vessels": 1}
interface = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=False)
interface_local_1 = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=True, local_range=[0.2, 0.7])

data_io = Data_IO(interface, data_path)
data_io_local_1 = Data_IO(interface_local_1, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
samples = interface.initialize(data_path)
print("All samples: " + str(sample_list))

sample_list_local_1 = data_io_local_1.get_indiceslist()
sample_list_local_1.sort()
samples_local_1 = interface_local_1.initialize(data_path)
print("All samples: " + str(sample_list_local_1))

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)

# Create and configure the Preprocessor class
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

# NETWORK
# Create the Neural Network model
model = Neural_Network(preprocessor=pp, architecture=unet_dense)
model_local_1 = Neural_Network(preprocessor=pp_local_1, architecture=unet_dense)

# model.reset_weights()

# start = timer()

model.load(GLOBAL_MODEL_NAME)
model_local_1.load(LOCAL_MODEL_NAME_1)

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


def cc(sample, slices):
    result = ccResult(sample, slices)
    result.remove_part('not-max', percentage=0.15)
    result.remove_part('dust', percentage=1.0)
    result.set_to_one()
    return result.labels

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
    if data.shape[1] == 512:
        s = 1e-3
    else:
        s = 0.1

    ax.scatter(x1,y1,z1,c='red',s=s)
    ax.set_box_aspect((0.429688, 0.429688, 0.625))

    try:
        os.mkdir(dir_name)
    except:
        pass
    views = {"Front":(0,0)}#, "Caudal_40":(40,0), "Cranial_55":(-55,0)} #, "RAO_60":(0,-60), "LAO_60":(0,60)}            
    for angle in views.keys():
        view = views[angle]
        ax.view_init(view[0],view[1])
        plt.savefig(dir_name+'/'+angle+'_'+filename,dpi=300)
        print(dir_name+'/'+angle+'_'+filename, flush=True)
        

def local_range(x, mode):
    if mode == "local_1":
        return x[int(x.shape[0]*0.2):int(x.shape[0]*0.7), 128:384, 128:384]
    return x[:,:,:]

def combine(pred_glob, pred_local_1):
    # start = timer()
    new_pred = pred_glob.copy()
    new_pred[int(new_pred.shape[0]*0.2):int(new_pred.shape[0]*0.7), 128:384, 128:384] = np.logical_or(
        local_range(new_pred, 'local_1'), pred_local_1
    )
    return new_pred


file_label = {
    "big_model":"big", "small_model_1":"small_1", "small_model_2":"small_2", "combine":"cb", "cc_after_combine":"cc",
    "global":"g", "local_1":"l1", "local_2":"l2"
    }

##
for mod in mode.keys():
    print(mode[mod], flush=True)
    mode_dir_name = mode[mod]
    try:
        os.mkdir(mode_dir_name)
    except:
        pass

    test_list = sample_list[mod::5]
    test_list_local_1 = sample_list_local_1[mod::5]
    test_list_num  = [mod+k*5 for k in range(len(test_list))]
    print(test_list_num)

    #start = timer()
    predictions = dict()
    predictions['big_model'] = model.predict(test_list, return_output=True, activation_output=False)
    predictions['small_model_1'] = model_local_1.predict(test_list_local_1, return_output=True, activation_output=False)
    #end = timer()
    #print(f"making {len(test_list)} predictions takes {end-start} s.",flush=True)

    dice_sum = dict()
    truth_flat = {"global":np.array([]), "local_1":np.array([])}
    pred_flat = dict()

    dice_list = dict()

    for model_name in ['big_model', 'combine', 'cc_after_combine']:
        pred_flat[model_name] = {"global":np.array([]), "local_1":np.array([])}
        dice_sum[model_name] = {"global":0.0, "local_1":0.0}

    for model_name in ["small_model_1"]:
        pred_flat[model_name] = np.array([])
        dice_sum[model_name] = 0.0

    # for each data
    for k in range(len(test_list)):
        ID = test_list_num[k]
        print(f"CVAI{ID+1}\t", flush=True)

        dice = {"big_model":{}, "small_model_1":0.0, "combine":{},"cc_after_combine":{}}

        n_slices = predictions['big_model'][k].shape[0]
        GT_global = data_io.sample_loader(sample_list[ID]).seg_data
        print(predictions['big_model'][k].shape, GT_global.shape, flush=True)
        GT_global = GT_global.reshape((GT_global.shape[0], 512,512))
        print(GT_global.shape, flush=True)


        # GT
        print('GT', flush=True)
        for label in ['global', 'local_1']:
            print(label, flush=True)
            GT = local_range(GT_global, label)
            print(GT_global.shape, GT.shape, flush=True)

            truth_flat[label] = np.concatenate((truth_flat[label], GT), axis=None)
            plot3d(GT, mode_dir_name+'/gt_'+file_label[label], 'gt_'+file_label[label]+'_'+str(ID+1)+'.png')

        preds = dict()
        preds['big_model'] = predictions['big_model'][k].copy()
        preds['small_model_1'] = predictions['small_model_1'][k].copy()
        print(preds['big_model'].shape, preds['small_model_1'].shape)
        preds['combine'] = combine(preds['big_model'], preds['small_model_1'])
        preds['cc_after_combine'] = cc(preds['combine'], n_slices)

        # small models
        for model_name in ["small_model_1"]:
            print(model_name, flush=True)
            label = 'local_'+model_name.split("_")[-1]
            print(label, flush=True)
            GT = local_range(GT_global, label)
            pred = preds[model_name]

            pred_flat[model_name] = np.concatenate((pred_flat[model_name], pred), axis=None)
            filename = f"pr_{file_label[model_name]}"
            plot3d(pred, f"{mode_dir_name}/{filename}", f"{filename}_{str(ID+1)}.png")

            dice[model_name] = simple_dice2(GT, pred)
            dice_sum[model_name] += dice[model_name]

        # big model
        for model_name in ['big_model', 'combine', 'cc_after_combine']:
            print(model_name, flush=True)
            for label in ['global', 'local_1']:
                print(label, flush=True)
                GT = local_range(GT_global, label)
                pred = local_range(preds[model_name], label)

                pred_flat[model_name][label] = np.concatenate((pred_flat[model_name][label], pred), axis=None)
                filename = f"pr_{file_label[model_name]}_{file_label[label]}"
                plot3d(pred, f"{mode_dir_name}/{filename}", f"{filename}_{str(ID+1)}.png")

                dice[model_name][label] = simple_dice2(GT, pred)
                dice_sum[model_name][label] += dice[model_name][label]

        # print dice scores
        dice_list["CVAI"+str(ID+1)] = dice
        print(dice, flush=True)

    # end of for each data

    with open(f'{mod}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["",  "big_model", "", "small_model", "combine", "", "cc_after_combine", ""])
        writer.writerow(["#", "global", "local_1", "local_1", "global", "local_1", "global", "local_1"])
        
        for ID in dice_list.keys():
            row_to_write = [ID]
            for model_name in dice_list[ID].keys():
                if model_name in ['small_model_1']:
                    row_to_write.append(dice_list[ID][model_name])
                else:
                    for label in dice_list[ID][model_name].keys():
                        row_to_write.append(dice_list[ID][model_name][label])
            print(row_to_write, flush=True)
            writer.writerow(row_to_write)

        row_to_write = ["Average"]
        for model_name in dice_sum.keys():
            if model_name in ['small_model_1']:
                row_to_write.append(dice_sum[model_name] / len(test_list))
            else:
                for label in dice_sum[model_name].keys():
                    row_to_write.append(dice_sum[model_name][label] / len(test_list))
        print(row_to_write, flush=True)
        writer.writerow(row_to_write)

        row_to_write = ["Flattened"]
        for model_name in pred_flat.keys():
            if model_name in ['small_model_1']:
                dice_score = simple_dice2(truth_flat['local_1'], pred_flat[model_name])
                row_to_write.append(dice_score)
            else:
                for label in pred_flat[model_name].keys():
                    dice_score = simple_dice2(truth_flat[label], pred_flat[model_name][label])
                    row_to_write.append(dice_score)
        print(row_to_write, flush=True)
        writer.writerow(row_to_write)

## end of for each mode
