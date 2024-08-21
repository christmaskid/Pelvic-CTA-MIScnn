# Modified by Alexander Shieh 2021/11/16
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import sys
print(sys.argv, flush=True)
# load model name
GLOBAL_MODEL_NAME = sys.argv[1] #"../../../../raid/data/b09401064/pelvic_models/unet_dense_30_v3_suf_8690.hdf5"
LOCAL_MODEL_NAME_1 = sys.argv[2] #"../../../../raid/data/b09401064/pelvic_models/unet_dense_30_v3_suf_local_new_best.hdf5"
mode = {4: 'test_5'} # 3: 'val'
data_path = sys.argv[3] #"../../../../raid/data/b09401064/pelvic_dataset_v2"
LOCAL_RANGE = eval(sys.argv[4]) #[0.2, 0.5]
parent_dir = sys.argv[5]

os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#output_path = "../../../../raid/data/b09401064/pelvic_dataset_v2_seg_pred"
#if not os.path.exists(output_path):
#    os.mkdir(output_path)

SAVE = False

DEBUG = True

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


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# self-defined libraries
from my_dicom_interface import My_DICOM_interface
# import boundary_loss as bd
from mycc3d_scipy import ccResult
import cc3d

# postprocessing
from skimage.morphology import closing, skeletonize_3d
from skimage import measure
import cc3d
from vessel_track import find_centerline_forest_3d, reconstruct_tree, flatten_forest
# from geometry import construct_mesh_and_get_contours
from test2 import save_html2

structure_dict = {"vessels": 1}
interface = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=False)
interface_local_1 = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
    annotation_tag="vessels", local=True, local_range=LOCAL_RANGE)
interface_IPA = My_DICOM_interface(structure_dict = {"IPA":1}, classes=2, annotation_tag="IPA", local=False)
# interface_local_2 = My_DICOM_interface(structure_dict = structure_dict, classes=2, 
#     annotation_tag="vessels", local=False, local_range=[0.0, 0.2])

data_io = Data_IO(interface, data_path)
data_io_local_1 = Data_IO(interface_local_1, data_path)
data_io_IPA = Data_IO(interface_IPA, data_path)

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


def recall(truth, pred):
    y_true_f = truth.flatten()
    y_pred_f = pred.flatten()
    tp = np.sum(y_true_f * y_pred_f)
    fp = np.sum((1-y_true_f) * y_pred_f)
    tn = np.sum((1-y_true_f) * (1-y_pred_f))
    fn = np.sum(y_true_f * (1-y_pred_f))
    return tp/(tp+fn)

def precision(truth, pred):
    y_true_f = truth.flatten()
    y_pred_f = pred.flatten()
    tp = np.sum(y_true_f * y_pred_f)
    fp = np.sum((1-y_true_f) * y_pred_f)
    tn = np.sum((1-y_true_f) * (1-y_pred_f))
    fn = np.sum(y_true_f * (1-y_pred_f))
    return tp/(tp+fp)

def IoU(truth, pred):
    y_true_f = truth.flatten()
    y_pred_f = pred.flatten()
    intersection = np.sum(truth * pred)
    return intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)


def cc(sample, slices):
    # sample[int(slices * 0.85):, :, :] = cc3d.largest_k(
    #     sample[int(slices * 0.85):, :, :], k=1, connectivity = 26, return_N = False
    # )
    # sample = cc3d.dust(sample, threshold=2000, connectivity = 26, in_place=True)
    # return sample
    result = ccResult(sample, slices)
    print("Before:", np.sum(result.labels>0), flush=True)
    result.remove_part('not-max', percentage=0.15)
    print("Remove not-max:", np.sum(result.labels>0), flush=True)
    result.remove_part('dust', percentage=1.0)
    print("Remove dust:", np.sum(result.labels>0), flush=True)
    result.set_to_one()
    print("Set to one", np.sum(result.labels==1), np.sum(result.labels>1), flush=True)
    return result.labels

from plot_3d import plot3d
        

def local_range(x, mode):
    if len(x.shape) == 4 and x.shape[-1]==1:
        x = x[:, :, :, 0]
    if mode == "local_1":
        return x[int(x.shape[0]*0.2):int(x.shape[0]*0.5), 128:384, 128:384]
    return x

def combine(pred_glob, pred_local_1, pred_local_2=None):
    # start = timer()
    pred_glob[int(pred_glob.shape[0]*0.2):int(pred_glob.shape[0]*0.5), 128:384, 128:384] = np.logical_or(
        local_range(pred_glob, 'local_1'), pred_local_1
    )
    return pred_glob


file_label = {
    "big_model":"big", "small_model_1":"small", "combine":"cb", "cc_after_combine":"cc",
    "global":"g", "local_1":"l1", "local_2":"l2"
    }
metric_label = {
    "dice": simple_dice2, "recall": recall, "IoU": IoU, "precision": precision,
    }

##
for mod in mode.keys():
    print(mode[mod], flush=True)
    mode_dir_name = parent_dir + "/" + mode[mod]
    try:
        os.mkdir(mode_dir_name)
    except:
        pass

    test_list = sample_list[mod::5]#[:1]
    test_list_local_1 = sample_list_local_1[mod::5]#[:1]
    # test_list_local_2 = sample_list_local_2[mod::5]
    test_list_num  = [mod+k*5 for k in range(len(test_list))]#[:1]
    
    print(test_list_num)

    if DEBUG:
        test_list = test_list[:2]
        test_list_local_1 = test_list_local_1[:2]
        test_list_num = test_list_num[:2]


    #start = timer()
    predictions = dict()
    predictions['big_model'] = model.predict(test_list, return_output=True, activation_output=False)
    predictions['small_model_1'] = model_local_1.predict(test_list_local_1, return_output=True, activation_output=False)
    #end = timer()
    #print(f"making {len(test_list)} predictions takes {end-start} s.",flush=True)


    metric_sum = {key: dict() for key in metric_label.keys()}
    #dice_sum = dict()
    truth_flat = {"global":np.array([]), "local_1":np.array([]), "local_2":np.array([])}
    pred_flat = dict()

    #dice_list = dict()
    metric_list = {key: dict() for key in metric_label.keys()}

    for model_name in ['big_model', 'combine', 'cc_after_combine']:
        pred_flat[model_name] = {"global":np.array([]), "local_1":np.array([])}
        #dice_sum[model_name] = {"global":0.0, "local_1":0.0}
        for key in metric_label.keys():
            metric_sum[key][model_name] = {"global": 0.0, "local_1": 0.0}

    for model_name in ["small_model_1"]:
        pred_flat[model_name] = {"local_1":np.array([])}
        #dice_sum[model_name] = {"local_1":0.0}
        for key in metric_label.keys():
            metric_sum[key][model_name] = {"local_1":0.0}

    # for each data
    for k in range(len(test_list)):
        # ID = test_list_num[k]
        ID = test_list[k]
        print(test_list[k], "\t", flush=True)

        #dice = {"big_model":{}, "small_model_1":{}, "combine":{},"cc_after_combine":{}}
        metrics = {key: {"big_model":{}, "small_model_1":{}, "combine":{},"cc_after_combine":{}}
                for key in metric_label.keys()}

        n_slices = predictions['big_model'][k].shape[0]
        GT_global = data_io.sample_loader(test_list[k]).seg_data
        GT_IPA = data_io_IPA.sample_loader(test_list[k]).seg_data


        # GT
        print('GT')
        for label in ['global', 'local_1']:
            print(label)
            GT = local_range(GT_global, label)#[0]
            print(GT_global.shape, GT.shape)

            truth_flat[label] = np.concatenate((truth_flat[label], GT), axis=None)
            # plot3d(GT, mode_dir_name+'/gt_'+file_label[label], 'gt_'+file_label[label]+'_'+ID+'.png')

        preds = dict()
        preds['big_model'] = predictions['big_model'][k].copy()
        preds['small_model_1'] = predictions['small_model_1'][k].copy()
        print(preds['big_model'].shape, preds['small_model_1'].shape)
        preds['combine'] = combine(preds['big_model'], preds['small_model_1'])
        preds['cc_after_combine'] = cc(preds['combine'], n_slices)
        preds['closing'] = closing(preds['cc_after_combine'])

        # small models
        #print(dice)
        for model_name in ["small_model_1"]:
            print(model_name)
            for label in ['local_1']:
                # label = 'local_'+model_name.split("_")[-1]
                print(label)
                GT = local_range(GT_global, label)
                pred = preds[model_name]

                pred_flat[model_name][label] = np.concatenate((pred_flat[model_name][label], pred), axis=None)
                filename = f"pr_{file_label[model_name]}"
                plot3d(pred, f"{mode_dir_name}/{filename}", f"{filename}_{test_list[k]}.png")

                #dice[model_name][label] = simple_dice2(GT, pred)
                #print(model_name, label, dice[model_name][label], flush=True)
                #dice_sum[model_name][label] += dice[model_name][label]

                for key in metric_label.keys():
                    metric_func = metric_label[key]
                    metrics[key][model_name][label] = metric_func(GT, pred)
                    print(model_name, label, metrics[key][model_name][label], flush=True)
                    metric_sum[key][model_name][label] += metrics[key][model_name][label]

                print(f"#### {test_list[k]}, {model_name}, {label}:", end=' ')
                for key in metric_label.keys():
                    print(f"{key}: {metrics[key][model_name][label]}, ", end='', flush=True)
                print()


        # big model
        for model_name in ['big_model', 'combine', 'cc_after_combine']:
            print(model_name)
            for label in ['global', 'local_1']:
                print(label)
                GT = local_range(GT_global, label)
                pred = local_range(preds[model_name], label)

                pred_flat[model_name][label] = np.concatenate((pred_flat[model_name][label], pred), axis=None)
                filename = f"pr_{file_label[model_name]}_{file_label[label]}"
                plot3d(pred, f"{mode_dir_name}/{filename}", f"{filename}_{ID}.png")

                #dice[model_name][label] = simple_dice2(GT, pred)
                #print(model_name, label, dice[model_name][label], flush=True)
                #dice_sum[model_name][label] += dice[model_name][label]

                for key in metric_label.keys():
                    metric_func = metric_label[key]
                    metrics[key][model_name][label] = metric_func(GT, pred)
                    print(model_name, label, metrics[key][model_name][label], flush=True)
                    metric_sum[key][model_name][label] += metrics[key][model_name][label]

                print(f"#### {test_list[k]}, {model_name}, {label}:", end=' ')
                for key in metric_label.keys():
                    print(f"{key}: {metrics[key][model_name][label]}, ", end='', flush=True)
                print()


        # print dice scores
        #dice_list[ID] = dice
        #print(dice, flush=True)
        for key in metric_label.keys():
            metric_list[key][ID] = metrics[key]
            print(key, metrics[key], flush=True)

        # save prediction
        if SAVE == True:
            sample = data_io.sample_loader(sample_list[ID])
            spacing = sample.get_extended_data()["spacing"]
            print(spacing, flush=True)
            affine = np.zeros((4,4))
            for i in range(3):
                affine[i][i] = spacing[i]
            print(affine, flush=True)
            nifti = nib.Nifti1Image(preds['closing'].astype(np.float64), affine)
            pred_file = f"CVAI{test_list[k]}" + ".nii.gz"
            # nib.save(nifti, os.path.join(output_path, pred_file))

        # calculate IPA recall
        IPA_recall = recall(GT_IPA, preds['cc_after_combine'])
        print(f"> {test_list[k]} IPA recall:", IPA_recall, flush=True)

        # post-processing
        print(preds['closing'].shape, flush=True)
        if not os.path.exists("htmls"):
            os.mkdir("htmls")
        save_html2(preds['closing'], f"htmls/{test_list[k]}_centerline.html", segmentation=False, centerline=True)


    # end of for each data
    save_dir = mode_dir_name

    for key in metric_label.keys():

        with open(f'{mode_dir_name}/{mod}_{key}.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["",  "big_model", "", "small_model", "combine", "", "cc_after_combine", ""])
            writer.writerow(["#", "global", "local_1", "local_1", "global", "local_1", "global", "local_1"])
        
            for ID in metric_list[key].keys():
                row_to_write = [ID]
                for model_name in metric_list[key][ID].keys():
                    for label in metric_list[key][ID][model_name].keys():
                        row_to_write.append(metric_list[key][ID][model_name][label])
                print(row_to_write, flush=True)
                writer.writerow(row_to_write)

            row_to_write = ["Average"]
            for model_name in metric_sum[key].keys():
                for label in metric_sum[key][model_name].keys():
                    row_to_write.append(metric_sum[key][model_name][label] / len(test_list))
            print(row_to_write, flush=True)
            writer.writerow(row_to_write)

            row_to_write = ["Flattened"]
            for model_name in metric_sum[key].keys():
                for label in pred_flat[model_name].keys():
                    metric_score = metric_label[key](truth_flat[label], pred_flat[model_name][label])
                    row_to_write.append(metric_score)
            print(row_to_write, flush=True)
            writer.writerow(row_to_write)

## end of for each mode
