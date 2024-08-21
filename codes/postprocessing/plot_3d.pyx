import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cimport numpy as cnp
import numpy as np
from timeit import default_timer as timer
import os

def plot3d(data:np.ndarray, str dir_name, str filename, double size=1e-3, axes_limits=None):
    cdef double start, end

    print("plot3d", filename,flush=True)  
    start = timer()
    
    z1, x1, y1 = np.where(data==1)
    print("Plot", len(z1), "dots.", flush=True)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1,y1,z1,c='red',s=size)
    ax.set_box_aspect((0.429688, 0.429688, 0.625))

    if axes_limits is not None:
        ax.set_zlim(axes_limits[0])
        ax.set_xlim(axes_limits[1])
        ax.set_ylim(axes_limits[2])

    try:
        os.mkdir(dir_name)
    except:
        pass
    views = {"Front":(0,0)}#, "RAO_60":(0,-60)}#, "LAO_60":(0,60), "Caudal_40":(40,0), "Cranial_55":(-55,0)}            
    for angle in views.keys():
        view = views[angle]
        ax.view_init(view[0],view[1])
        plt.tight_layout()
        plt.savefig(dir_name+'/'+angle+'_'+filename,dpi=300)
        print(dir_name+'/'+angle+'_'+filename, flush=True)
        
    end = timer()
    print(f"spent {end-start} s.",flush=True)

    plt.close()

def plot3d_multi(list data_list, str dir_name, str filename, list color_list, list radius_list, list opacity_list, axes_limits=None):
    cdef double start, end
    cdef str color
    cdef cnp.ndarray[cnp.int64_t, ndim=3] data
    cdef double alpha, radius

    print("plot3d multi", filename, flush=True)  
    start = timer()
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for data, color, radius, alpha in zip(data_list, color_list, radius_list, opacity_list): 
        z1, x1, y1 = np.where(data==1)
        print("Plot", len(z1), "dots.", flush=True)
        ax.scatter(x1,y1,z1,c=color,s=radius, alpha=alpha)

    ax.set_box_aspect((0.429688, 0.429688, 0.625))

    if axes_limits is not None:
        ax.set_zlim(axes_limits[0])
        ax.set_xlim(axes_limits[1])
        ax.set_ylim(axes_limits[2])

    try:
        os.mkdir(dir_name)
    except:
        pass
    views = {"Front":(0,0)}#, "RAO_60":(0,-60) #, "LAO_60":(0,60), "Caudal_40":(40,0), "Cranial_55":(-55,0)}            
    for angle in views.keys():
        view = views[angle]
        ax.view_init(view[0],view[1])
        plt.savefig(dir_name+'/'+angle+'_'+filename,dpi=300)
        print(dir_name+'/'+angle+'_'+filename, flush=True)  

    end = timer()
    print(f"spent {end-start} s.",flush=True)
    plt.close()
