# Original author: Hsiang-Chin Chien
# Modified by Yu-Tong Cheng: a 3D-interpolation version

import numpy as np
from numpy import dot, cross
from numpy.linalg import inv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import trange
import csv

#from mayavi import mlab

from scipy.io import savemat,loadmat

import os

from itertools import accumulate, compress
from scipy.spatial import Delaunay
import pydicom
import struct
from scipy import interpolate

from pydicom.data import get_testdata_file
from pydicom import dcmread
from pydicom.datadict import dictionary_VR
from hexdump import hexdump
import netfleece
import json

def explore_AVA_dicom(dir1):
    diff = []
    cnt = 0
    files = dict()
    for filename in os.listdir(dir1):
        try:
            ds1 = pydicom.dcmread(dir1+filename)
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
        print("Parse", i)
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

            id_list = [obj_id, maxd_pt1_id, maxd_pt2_id, mind_pt1_id, mind_pt2_id, centroid_id, contour_id]
            idref_list = []
            #print(id_list)
            #for cid in id_list[1:]:
            #    idref_list.append(obj_dict[cid]['Values'][0]['IdRef'])
            #print(idref_list)

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
                'mind1': getPt(mind_pt1_id), 'mind2': getPt(mind_pt2_id)#,
                #'IdList': id_list, "IdRefList": idref_list
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

    #print(tissue_path)
    #print(ds)
    #print(boundery)
    #input()

    # m=ds.Rows
    # n=ds.Columns
    # zb=ds.get_item(key=(0x0045, 0x1001)).value
    # z=int(zb[0:1].hex(),16)
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

    for i in trange(len(ori_array)-1):
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

if __name__ == '__main__':
    
    """
    this is a example for single vessel annotation
    from MPR to segmentation annotation
    """

    anno_path = 'S138490\\S5160\\' # annotation file: folder of dicom files (I10, I20, ..., etc)
    first_slice_path = "datasets\\V3_new_30\\0347867_0347867-20220316T050045Z-001\\0347867_0347867\\t0157606419\\0.625mm_CTA\\IM-0001-0521.dcm" 
    #'first ct slice path' is actually the 'last' one in pelvic datasets since the last slice's z-coordination of origin is the lowest.
    ptid = "0347867"

    #### Get pixelspacing and dimension
    spx,spy,spz,origin,dim = get_original_coordination_and_volume(first_slice_path)
    print(spx, spy, spz, origin, dim)
    
    #### Get info on files
    cnt, files = explore_AVA_dicom(anno_path)
    print(files)
    targets = ["L_IPA", "R_IPA"]
    Xs_, Ys_, Zs_ = [], [], []

    for vessel_name , filename in files.items():
        print(vessel_name, filename)
        if vessel_name not in targets:
            continue

        results = parse_single_vessel(anno_path+filename)
        pts, pts_, obj_dict, centerlines = results[0] # 0: inner wall, 1: outer wall
        #print(len(pts))

        center_points = np.array(centerlines[0::3])
        normalVector = np.array(centerlines[1::3])
        tangentVector = np.array(centerlines[2::3])

        obj_id_list, contour_list = get_contour_list(obj_dict)
        #print(len(contour_list))

        contours = getDict("contour", contour_list)
        #print(contours[0])
        # A list of contours. Each element contains a list of 3D coordinations of points on the contour.
        
        ## Fill each contour with the same amount of points
        circ_contours = []
        for i in trange(len(contours)):
            circ_contours.append(interpolate_circle(contours[i]))

        #print(len(circ_contours), len(center_points), len(normalVector), len(tangentVector))
        
        ## Interpolate new contours in between
        intp_num = 1
        for n in range(intp_num):
            circ_contours = interpolate_vertical(circ_contours)
            center_points = interpolate_vertical(center_points)
            normalVector = interpolate_vertical(normalVector)
            tangentVector = interpolate_vertical(tangentVector)
        #print(len(circ_contours), len(center_points), len(normalVector), len(tangentVector))

        ## Fill each contour into a solid circle
        plane_contours = []
        for i in trange(len(circ_contours)):
            plane_contours.append(fill_circles(circ_contours[i], center_points[i]))

        ## Get the set of all points
        point_set = plane_contours[0].copy()
        for i in trange(len(plane_contours)):
            point_set = np.concatenate((point_set, plane_contours[i]), axis=0)
        #print(point_set.shape)

        #### Calculate dicom coordinate to CT coordinate
        point_set = org_coord_trans(point_set, first_slice_path)

        #### Generate segmentation annotation
        #ds = pydicom.dcmread(anno_path+"I10")
        #ct_size = ds[0x07a1, 0x1007].value
        #print(ct_size)

        #print(len(point_set))
        point_set = list(set((int(pt[0]),int(pt[1]),int(pt[2])) for pt in point_set))
        #print(len(point_set))
        #print(point_set)

        Xs, Ys, Zs = [], [], []
        for i in trange(len(point_set)):
            x=point_set[i][0]
            y=point_set[i][1]
            z=point_set[i][2]
            Xs.append(x); Ys.append(y); Zs.append(z)

        Xs_ += Xs
        Ys_ += Ys
        Zs_ += Zs
        
    print(len(Xs_), len(Ys_), len(Zs_))
    
    #### Save all the points
    all_pts = []
    for x,y,z in zip(Xs_, Ys_, Zs_):
        all_pts.append([z,y,x])
    with open(ptid+'_seg_new_IPA'+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_pts)

    #### Plot the new labels with original ones with plotly
    ## Plotly does not work on scatter plots with more than 1e6 points, so we plot the overlapped part only.
    colors = ["blue"] * len(Xs_)
    mx = min(Xs_); Mx = max(Xs_)
    my = min(Ys_); My = max(Ys_)
    mz = min(Zs_); Mz = max(Zs_)
    print(mx, Mx, my, My, mz, Mz)

    sample = pd.read_csv("datasets\\V3_new_30_csv\\0347867_seg_new.csv",header=None).to_numpy()
    x, y, z = [], [], []
    for i in trange(len(sample)):
        pt = sample[i]
        if mz < pt[0] and pt[0] < Mz \
            and mx < pt[2] and pt[2] < Mx \
            and my < pt[1] and pt[1] < My:
            z.append(pt[0]); x.append(pt[2]); y.append(pt[1])

    Xs_ += x; Ys_ += y; Zs_ += z
    colors += ["red"] * len(x)

    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.offline as pyo
    
    marker_data = go.Scatter3d(
        x=Xs_, 
        y=Ys_, 
        z=Zs_, 
        #marker=go.scatter3d.Marker(size=2), 
        marker=dict(
            color=colors,
            size=0.5
        ),
        opacity=0.8, 
        mode='markers'
    )
    fig=go.Figure(data=marker_data)
    fig.update_layout(             
                title_text="IPA",
                scene=dict(
                    aspectmode='data',
                    camera = dict(projection=dict(type="orthographic"))
                    )
        )
    fig.write_html(ptid+'_IPA.html')#, auto_open=True)
