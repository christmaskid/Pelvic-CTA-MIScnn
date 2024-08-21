import os
import numpy as np
from numpy import dot, cross
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from itertools import accumulate, compress
from scipy import interpolate
from scipy.io import savemat,loadmat
from scipy.spatial import Delaunay
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
from pydicom.datadict import dictionary_VR
from hexdump import hexdump
import netfleece
import json
from tqdm import trange
import csv
import shutil
import sys, getopt

def decode_file(dim1, raw_bytes2, filename, file_type='v'):
    cnt = 0
    clue2 = raw_bytes2[dim1[2]*1024:]
    all_pts = []
    
    for j in range(dim1[2]):
        pts = []
        for idx, bt in enumerate(raw_bytes2[j*1024:(j+1)*1024]):
            pq = []
            n1 = int(bt)
            if n1 == 0:
                continue
            x1 = int(idx/2)
            for i in range(0, n1):
                y1 = int(clue2[cnt+i*2])
                y2 = int(clue2[cnt+i*2+1])

                if y2 % 2 == 0:
                    pq.append((y1, x1, y2))
                else:
                    pq.append((y1+256, x1, y2))
            cnt += n1*2

            flag = 0
            last = -1
            for pp in pq:
                if(pp[2] >= 128):
                    if flag == 1:
                        for k in range(last+1, pp[0]):
                            pts.append((k, pp[1]))
                    else:
                        flag = 1
                    pts.append((pp[0], pp[1]))
                    last = pp[0]
                else:
                    if flag == 1:
                        for k in range(last+1, pp[0]):
                            pts.append((k, pp[1]))
                    pts.append((pp[0], pp[1]))
                    for i in range(int(pp[2]/4)):
                        pts.append((pp[0]+i+1, pp[1]))
                    flag = 0

        for pt in pts:
            all_pts.append((j, pt[1], pt[0]))
        
    if file_type == 'v':
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_pts)


def explore_AVA_dicom(dir1):
    diff = []
    cnt = 0
    files = dict()
    for filename in os.listdir(dir1):
        try:
            ds1 = pydicom.dcmread(os.path.join(dir1,filename))
            #print(filename)
            #print(ds1[0x0020, 0x4000])
            #print(ds1[0x00e1, 0x1047])
            #print(ds1[0x07a1, 0x1012])
            vessel_name = ds1[0x0020, 0x4000].value
            if vessel_name != 'Unnamed':
                files[vessel_name] = filename
                cnt += 1
        except:
            pass
    
    return cnt, files

def parse_single_vessel(filename):
    ds1 = pydicom.dcmread(filename)
    #print("###Original Binary###")
    # hexdump(ds1[0x00e1, 0x1047].value[4:100])

    newFile = open(filename+".bin", "wb")
    if ds1[0x00e1, 0x1047].value[-1] == 11:
        newFile.write(bytes(ds1[0x00e1, 0x1047].value[4:]))
    else:
        newFile.write(bytes(ds1[0x00e1, 0x1047].value[4:-1]))
    
    newFile.close()
    infile = open(filename+".bin", 'rb')
    parsed = netfleece.parseloop(infile, decode=False, expand=False,
                   backfill=False, crunch=False, root=False)

    with open(filename+".json", 'w') as outf:
        outf.write(json.dumps(parsed))

    results = []
    for i in range(len(parsed)):
        # print("Parse", i)
        parse = parsed[i]
        obj_dict = {}
        for obj in parse:
            if "ClassInfo" in obj:
                obj_dict[obj["ClassInfo"]["ObjectId"]] = obj
            elif "ArrayInfo" in obj:
                obj_dict[obj["ArrayInfo"]["ObjectId"]] = obj
            elif "ObjectId" in obj:
                obj_dict[obj["ObjectId"]] = obj

        pts = []
        pts_ = []

        for o in obj_dict:
            try:
                if "ArrayInfo" in obj_dict[o]:
                    if len(obj_dict[o]['Values']) == 4:
                        pts_.append(obj_dict[o]['Values'])
                    elif len(obj_dict[o]['Values']) == 3:
                        pts.append(obj_dict[o]['Values'])
            except:
                pass

        raw_list = ds1[0x07a1, 0x1012].value
        raw_list = raw_list[:]
        new_list = []
        for i in range( int(len(raw_list)/3) ):
            new_list.append([raw_list[3*i], raw_list[3*i+1], raw_list[3*i+2]])

        results.append([pts, pts_, obj_dict, new_list])

    return results

def get_contour_list(obj_dict):
    contour_list = []
    cnt = 0
    obj_id_list = []
    for obj_id in obj_dict:
        obj = obj_dict[obj_id]
        if 'MetadataId' in obj and obj['MetadataId'] == 6 \
                or 'ClassInfo' in obj and obj['ClassInfo']['Name']=="AVA.Model.AVAContour": # first contour
            obj_id_list.append(obj_id)
            area = obj['Values'][0]
            effective_diameter = obj['Values'][1]
            #print(area, effective_diameter)
            maxd_pt1_id = obj['Values'][2]['IdRef']
            maxd_pt2_id = obj['Values'][3]['IdRef']
            mind_pt1_id = obj['Values'][4]['IdRef']
            mind_pt2_id = obj['Values'][5]['IdRef']
            centroid_id = obj['Values'][6]['IdRef']
            contour_id = obj['Values'][10]['IdRef']

            def getRefList(cid):
                pt_list = []
                id_list = []
                #print(cid, ">>", obj_dict[cid])
                for pt in obj_dict[obj_dict[cid]['Values'][0]['IdRef']]['Values']:
                    #print(pt)
                    if('IdRef' in pt):
                        idref = obj_dict[pt['IdRef']]['Values'][0]['IdRef']
                        id_list.append(idref)
                        pt_list.append(obj_dict[idref]['Values'])
                #return id_list, pt_list
                return pt_list

            def getPt(cid):
                #print(cid, ">>", obj_dict[cid])
                #print(obj_dict[obj_dict[cid]['Values'][0]['IdRef']])
                pt = obj_dict[obj_dict[cid]['Values'][0]['IdRef']]['Values']
                return pt

            tmp_dict = {'area': area, 'diameter': effective_diameter, 
                'centroid': getPt(centroid_id), 'contour': getRefList(contour_id),
                'maxd1': getPt(maxd_pt1_id), 'maxd2': getPt(maxd_pt2_id),
                'mind1': getPt(mind_pt1_id), 'mind2': getPt(mind_pt2_id)
                }

            contour_list.append(tmp_dict)

    return obj_id_list, contour_list

def get_original_coordination_and_volume(tissue_path):
    
    coordinates=[]
    deltaA=[]
    
    ds = pydicom.dcmread(tissue_path)
    
    boundery = ds.get_item(key=(0x0020, 0x0032)).value.decode('ISO_IR 100','strict')
    
    for x in (boundery.split('\\')):
        coordinates.append(float(x))

    dim=[0,0,0]

    deltax=ds.PixelSpacing[0]
    deltay=ds.PixelSpacing[1]
    deltaz=ds.SliceThickness
    
    origin=[coordinates[0],coordinates[1],coordinates[2]]

    return deltax, deltay, deltaz, origin, dim

def org_coord_trans(xyz_point_set,volume_path):
    
    dx,dy,dz,origin,dim=get_original_coordination_and_volume(volume_path)
    xyz_point_set[:,0]=np.asarray((xyz_point_set[:,0]-origin[0])/(dx))+0
    xyz_point_set[:,1]=np.asarray((xyz_point_set[:,1]-origin[1])/(dy))+0
    xyz_point_set[:,2]=np.asarray((xyz_point_set[:,2]-origin[2])/(dz))+1
    
    return xyz_point_set

def mask_gen(trans_point,dim):
    
    m=dim[0]
    n=dim[1]
    z=dim[2]

    point_mask=np.zeros((m,n,z),np.uint8)
    
    for i in trange(0,len(trans_point)):
        x=int(trans_point[i,1])
        y=int(trans_point[i,0])
        z=int(trans_point[i,2])
        #print(x,y,z)
        point_mask[x,y,z]=255
        
    return point_mask

def getDict(key, clist):
    a = []
    for obj in clist: a.append(obj[key])
    return a

def getDictArr(key, clist):
    a = []
    for obj in clist: a += obj[key]
    return a

def interpolate_vertical(ori_array):
    intp_list = [np.array(ori_array[0])]

    for i in range(len(ori_array)-1):
        plane_1 = np.array(ori_array[i])
        plane_2 = np.array(ori_array[i+1])
        new_plane = (plane_1 + plane_2)/2
        intp_list.append(new_plane)
        intp_list.append(plane_2)

    return intp_list

def interpolate_circle(contour_points):
    x = [pt[0] for pt in contour_points]
    y = [pt[1] for pt in contour_points]
    z = [pt[2] for pt in contour_points]

    # ref: Hsiang-chin chien
    x = np.r_[x, x[0]] # the first and the last point must be the same to form a ring
    y = np.r_[y, y[0]]
    z = np.r_[z, z[0]]

    # https://stackoverflow.com/questions/18962175/spline-interpolation-coefficients-of-a-line-curve-in-3d-space
    tck, u = interpolate.splprep([x,y,z],k=3,s=0,per=True)
    xk,yk,zk = interpolate.splev(np.arange(0, 1.01, 0.003),tck)
    circ_points = np.zeros([xk.shape[0],3])
    circ_points[:,0] = xk
    circ_points[:,1] = yk
    circ_points[:,2] = zk

    return circ_points

def fill_circles(circ_points, center_point):
    plane_points = circ_points.copy()
    n=100
    scale = 1
    for i in range(n):
        scale -= 1/n
        new_circ = (circ_points.copy()-center_point)*scale+center_point
        plane_points = np.concatenate((plane_points,new_circ),axis=0)
    return plane_points

def convertArray1To3(sample):
    length = len(sample)
    z1 = [sample[i][0] for i in range(length)]
    x1 = [sample[i][1] for i in range(length)]
    y1 = [sample[i][2] for i in range(length)]
    print("Finish converting files.")
    return [x1,y1,z1]

def extract_contours(anno_path, first_slice_path, ptid, targets, out_file):

    #### Get pixelspacing and dimension
    spx,spy,spz,origin,dim = get_original_coordination_and_volume(first_slice_path)
    print(spx, spy, spz, origin, dim)
    
    ####
    cnt, files = explore_AVA_dicom(anno_path)
    print(files)
    Xs_, Ys_, Zs_ = [], [], []

    for vessel_name , filename in files.items():
        print(vessel_name, filename)
        if vessel_name not in targets:
            continue

        results = parse_single_vessel(os.path.join(anno_path, filename))
        pts, pts_, obj_dict, centerlines = results[0] # 0: inner points, 1: outer points

        center_points = np.array(centerlines[0::3])
        normalVector = np.array(centerlines[1::3])
        tangentVector = np.array(centerlines[2::3])

        obj_id_list, contour_list = get_contour_list(obj_dict)
        contours = getDict("contour", contour_list)
        
        circ_contours = []
        for i in range(len(contours)):
            circ_contours.append(interpolate_circle(contours[i]))

        intp_num = 1
        for n in range(intp_num):
            circ_contours = interpolate_vertical(circ_contours)
            center_points = interpolate_vertical(center_points)
            normalVector = interpolate_vertical(normalVector)
            tangentVector = interpolate_vertical(tangentVector)

        plane_contours = []
        for i in range(len(circ_contours)):
            plane_contours.append(fill_circles(circ_contours[i], center_points[i]))

        point_set = plane_contours[0].copy()
        for i in trange(len(plane_contours)):
            point_set = np.concatenate((point_set, plane_contours[i]), axis=0)

        #### Calculate dicom coordinate to CT coordinate
        point_set = org_coord_trans(point_set, first_slice_path)

        #print(len(point_set))
        point_set = list(set((int(pt[0]),int(pt[1]),int(pt[2])) for pt in point_set))

        Xs, Ys, Zs = [], [], []
        for i in range(len(point_set)):
            x=point_set[i][0]
            y=point_set[i][1]
            z=point_set[i][2]
            Xs.append(x); Ys.append(y); Zs.append(z)

        Xs_ += Xs
        Ys_ += Ys
        Zs_ += Zs
        
    #print(len(Xs_), len(Ys_), len(Zs_))
    
    if len(Xs_)==0:
        print(f"No file is found in {anno_path} for target \'{targets}\'.")
        return
    else:
        print(f"{len(Xs_)} points have been generated and saved to {out_file}.")
        all_pts = sorted([[z,y,x] for x,y,z in zip(Xs_, Ys_, Zs_)])
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_pts)

def rename_dicom(path):
    for fn in os.listdir(path):
        if fn=='DIRFILE':
            os.remove(os.path.join(path, fn))
        else:
            new_fn = 'IM-0001-'+fn[1:-1].zfill(4)+'.dcm'
            os.rename(os.path.join(path,fn), os.path.join(path, new_fn))

def convertArray1To3(sample):
    print('converting sample...', end='\r')
    length = len(sample)
    z1 = np.array([sample[i][0] for i in range(length)])
    x1 = np.array([sample[i][1] for i in range(length)])
    y1 = np.array([sample[i][2] for i in range(length)])
    return [x1,y1,z1]

def plot3d(in_file, out_file, color='red', s=1e-3):
    if not os.path.exists(in_file):
        print(f"File {in_file} does not exist. Please check.")
        return
    seg = pd.read_csv(in_file, header=None).to_numpy()
    data = convertArray1To3(seg)

    print(f"plotting 3d to {out_file}...", end='\r')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[0],data[1],data[2],c=color, s=s)
    ax.set_box_aspect((0.429688, 0.429688, 0.625))
    ax.view_init(0,0)
    plt.savefig(out_file,dpi=300)
    print(f"The scatter plot is saved to {out_file}.")

def plot_combine_3d(in_files, out_file, colors, s=None):
    if len(in_files) != len(colors):
        raise ValueError("Numbers of in_files and colors provided are not the same.")
    if s is None:
        s = [1e-3 for file in in_files]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for i in range(len(in_files)):
        in_file = in_files[i]
        if not os.path.exists(in_file):
            print(f"File {in_file} does not exist. Please check.")
            return
        seg = pd.read_csv(in_file, header=None).to_numpy()
        data = convertArray1To3(seg)
        ax.scatter(data[0],data[1],data[2],c=colors[i],s=s[i])

    ax.set_box_aspect((0.429688, 0.429688, 0.625))
    ax.view_init(0,0)
    plt.savefig(out_file,dpi=300)

def fileType(directory, target):
    files = os.listdir(directory)
    if len(files)==0:
        return False

    ds = pydicom.dcmread(os.path.join(directory, files[0]))
    if(ds[0x0008, 0x0008].value[:len(target)]==target):
        return True
    return False

def isSeries(directory):
    return fileType(directory, ['ORIGINAL', 'PRIMARY', 'AXIAL'])

def isAnno(directory):
    return fileType(directory, ['DERIVED', 'SECONDARY'])


def explore_AVA_dicom_centerlines(anno_path, first_slice_path, ptid, targets, out_file):
    
    #### Get pixelspacing and dimension
    spx,spy,spz,origin,dim = get_original_coordination_and_volume(first_slice_path)
    print(spx, spy, spz, origin, dim)
    
    ####
    cnt, files = explore_AVA_dicom(anno_path)
    print(files)
    Xs_, Ys_, Zs_ = [], [], []

    for vessel_name , filename in files.items():
        #print(vessel_name, filename)
        if vessel_name not in targets:
            continue

        ds = dcmread(os.path.join(anno_path, filename))
        raw_list = ds[0x07a1, 0x1012].value[:]
        new_list = []
        for i in range( int(len(raw_list)/3) ):
            new_list.append([raw_list[3*i], raw_list[3*i+1], raw_list[3*i+2]])
        center_points = new_list[0::3]
        normal_vectors = new_list[1::3]

        point_set = np.array(center_points)

        #### Calculate dicom coordinate to CT coordinate
        point_set = org_coord_trans(point_set, first_slice_path)

        #print(len(point_set))
        point_set = list(set((int(pt[0]),int(pt[1]),int(pt[2])) for pt in point_set))

        Xs, Ys, Zs = [], [], []
        for i in range(len(point_set)):
            x=point_set[i][0]
            y=point_set[i][1]
            z=point_set[i][2]
            Xs.append(x); Ys.append(y); Zs.append(z)

        Xs_ += Xs
        Ys_ += Ys
        Zs_ += Zs
        
    #print(len(Xs_), len(Ys_), len(Zs_))
    
    if len(Xs_)==0:
        print(f"No file is found in {anno_path} for target \'{targets}\'.")
        return
    else:
        print(f"{len(Xs_)} points have been generated and saved to {out_file}.")
        all_pts = sorted([[z,y,x] for x,y,z in zip(Xs_, Ys_, Zs_)])
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_pts)


if __name__=='__main__':

    GEN_CSV = True      # Whether to generate .csv files of segmentation results
    GEN_SEG_CSV = True  # generate 'vessels' segmentation
    GEN_IPA_CSV = True  # convert IPA contours to segmentation
    GEN_CTP_CSV = False # Generate centerlines (centerpoints)
    GEN_GRAPH = True    # Whether to generate graphs of segmentation
    GEN_DATASET = True  # Whether to save files to the dataset of specified folder path

    dir_name = input("Enter directory of series: ")

    ## Generate dataset ##
    if GEN_DATASET:
        if GEN_CSV == False:
            print(f"Illegal setting. GEN_CSV must be set to True while GEN_DATASET is True.")
            sys.exit()
        destination = input("Enter directory of file saving destination (dataset): ")
        if not os.path.exists(destination):
            os.mkdir(destination)

    fn_id_list = []

    ## Parsing files ##
    list_of_data = sorted(os.listdir(dir_name))
    for dirc in list_of_data:
        print("="*50)
        dir2 = os.path.join(dir_name, dirc)
        subdirs = [dir1 for dir1 in os.listdir(dir2) if len(dir1.split('.'))==1 and dir1[0]=='S']
        print(dir2, subdirs)
        if len(subdirs)!=2:
            raise ValueError(f"The number of subdirectories in {dir2} is {len(subdirs)}, but the required number is 2.")

        series_path, anno_path = None, None
        for i in range(2):
            if(isSeries(os.path.join(dir2, subdirs[i]))):
                if series_path is None:
                    series_path = subdirs[i]
                else:
                    raise Exception(f"More than 2 image series: {series_path}, {subdirs[i]} in {dir2}.")
            elif(isAnno(os.path.join(dir2, subdirs[i]))):
                if anno_path is None:
                    anno_path = subdirs[i]
                else:
                    raise Exception(f"More than 2 annotations: {anno_path}, {subdirs[i]} in {dir2}.")

        if series_path is None:
            raise Exception(f"No image series path is found in {dir2}.")
        if anno_path is None:
            raise Exception(f"No annotation path is found in {dir2}.")
        series_path = os.path.join(dir2, series_path)
        anno_path = os.path.join(dir2, anno_path)
        print('image:', series_path)
        print('label:', anno_path)

        first_file = pydicom.dcmread(os.path.join(anno_path, os.listdir(anno_path)[0]))
        ptid = first_file[0x0010, 0x0020].value
        pc_data = first_file[0x0008, 0x0020].value
        print(ptid, pc_data)
        key_id = ptid+'_'+pc_data
        fn_id_list.append([dirc, ptid+' ('+pc_data+')'])

        SEG_FILENAME = key_id+'_seg_new'+'.csv'
        IPA_FILENAME = key_id+'_seg_new_IPA'+'.csv'
        CTP_FILENAME = key_id+'_ctp.csv'

        if GEN_CSV:
            ## Parse files and get 'vessels' segmentation ##
            if GEN_SEG_CSV:
                for filename in os.listdir(os.path.join(dir_name, dir2, anno_path)):
                    try:                    
                        ds1 = dcmread(os.path.join(dir_name, dir2, anno_path, filename))                
                        if ds1[0x0020, 0x4000].value == 'Vessels':
                            dim1 = ds1[0x07a1, 0x1007].value
                            print(dim1)
                            raw_bytes2 = ds1[0x07a1, 0x1009].value[:]
                            decode_file(dim1, raw_bytes2, SEG_FILENAME)
                    except:
                        continue

            last_slice = os.path.join(series_path,"I"+str(len(os.listdir(series_path))-1)+"0")
            print(last_slice)

            ## Extract contours and convert them to segmentation ##
            if GEN_IPA_CSV:
                extract_contours(anno_path, last_slice, ptid, ["L_IPA", "R_IPA", "L-IPA", "R-IPA"], IPA_FILENAME)

            ## Extract centerpoints ##
            if GEN_CTP_CSV:
                explore_AVA_dicom_centerlines(anno_path, last_slice, ptid, ["L_IPA", "R_IPA", "L-IPA", "R-IPA"], CTP_FILENAME)

        if GEN_DATASET:
            if len(os.listdir(destination))==0:
                last_num = 1
            else:
                last_num = int(sorted(os.listdir(destination))[-1].split("_")[-1])+1
            code_name = 'CVAI_Train_'+str(last_num).zfill(2)
            fn_id_list[-1].append(code_name)
            print(f"Saving to dataset {code_name} ...", end='\r')

            folder_name = os.path.join(destination, code_name)
            seg_folder = os.path.join(folder_name, 'vessels.'+ptid)
            IPA_folder = os.path.join(folder_name, 'IPA.'+ptid)
            os.mkdir(folder_name)
            os.mkdir(seg_folder)
            os.mkdir(IPA_folder)

            shutil.copytree(series_path, os.path.join(folder_name, '0.625mm CTA'))
            shutil.copyfile(SEG_FILENAME, os.path.join(seg_folder, SEG_FILENAME))
            shutil.copyfile(IPA_FILENAME, os.path.join(IPA_folder, IPA_FILENAME))

            ## Rename and clean up dicom image files
            rename_dicom(os.path.join(folder_name, '0.625mm CTA'))

        ## Generate graphs of ground truth ##
        if GEN_GRAPH:
            if GEN_SEG_CSV == True:
                plot3d(SEG_FILENAME, SEG_FILENAME.split(".")[0]+'.jpg', color='red', s=1e-4)
            if GEN_IPA_CSV == True:
                plot3d(IPA_FILENAME, IPA_FILENAME.split(".")[0]+'.jpg', color='blue', s=1e-3)
            if GEN_SEG_CSV == True and GEN_IPA_CSV == True:
                plot_combine_3d([SEG_FILENAME, IPA_FILENAME], key_id+'_seg_cb.jpg', colors=['red', 'blue'], s=[1e-4, 1e-3])
            if GEN_CTP_CSV == True:
                plot3d(CTP_FILENAME, CTP_FILENAME.split(".")[0]+'.jpg', color='green', s=1)

    ## Print out patient id for each file ##
    print("="*50)
    fn_id_list.sort()
    for item in fn_id_list:
        for n in item:
            print(f"{n}\t",end='')
        print(end='\n')
