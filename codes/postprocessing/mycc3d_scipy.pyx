from scipy import ndimage
cimport numpy as cnp
import numpy as np
import cc3d
import time

cdef class ccResult:
    cdef public:
        int slices, N
        object labels, components_volume

    def __init__(self, cnp.ndarray sample, int slices):
        self.slices = slices
        self.labels = np.zeros((sample.shape[0], sample.shape[1], sample.shape[2]))

        self.find_cc(sample, slices)

    cpdef find_cc(self, cnp.ndarray sample, int slices):
        cdef double start, end
        cdef object labels_out
        cdef list tmp_vols
        
        start = time.time()

        print(f"{slices} slices.",flush=True)
        print(sample.shape[0], sample.shape[1], sample.shape[2], flush=True)
        # total_volume = (labels_in==1).sum()

        # labels_out, N = ndimage.label(sample, ndimage.generate_binary_structure(3,3))
        labels_out, N = cc3d.connected_components(
            sample, connectivity=26, return_N=True,
        )
        print(f"{N} components in total.",flush=True)
        
        end = time.time()
        print(f"cc3d: {end-start} s.",flush=True)

        self.labels = labels_out
        self.N = N
        stats = cc3d.statistics(labels_out)
        self.components_volume = stats["voxel_counts"]
        # print("voxel_counts:", self.components_volume, flush=True)
        del stats, labels_out

    cpdef remove_part(self, str mode, double percentage=1.0, double threshold=2000):
        cdef:
            double start = time.time()
            tuple ori_shape
            int slices, see_range

        ori_shape = self.labels.shape
        slices = self.slices
        see_range = int(percentage * slices)
        print(f"{percentage*100}% of sample contains {see_range} slices.", flush=True)
        print(mode, flush=True)

        if mode == 'dust':
            self.remove_dusts(threshold=threshold), 
        elif mode == 'not-max':
            self.remove_not_max(see_range=int(self.slices * percentage))
        
        cdef double end = time.time()
        print(mode, end-start, '(s)', flush=True)

    cpdef set_to_one(self):
        self.labels = (self.labels>0).astype(int)

    cpdef remove_dusts(self, double threshold): # revised from dust
        # remove_mask = np.zeros(self.labels.shape)
        # for i in range(1, self.N+1):
        #     # print(i, self.components_volume[i])
        #     if self.components_volume[i] < threshold:
        #         remove_mask += (self.labels==i)#.astype(int)
        #     else:
        #         break
        # self.labels *= (remove_mask==0)#.astype(int)
        self.labels = cc3d.dust(self.labels, threshold=threshold, connectivity=26, in_place=True)

    def sort_key(self, x):
        return x[1]

    cpdef remove_not_max(self, int see_range): # revised from largest_k
        cdef:
            int max_i
            list sortedVolumeIndice
            object labels_sub

        sortedVolumeIndice = sorted( [(k+1,v) for k,v in enumerate(self.components_volume[1:])], key=self.sort_key)
        max_i = sortedVolumeIndice[-1][0]
        # print(self.components_volume, sortedVolumeIndice, flush=True)
        print("Max i:", max_i, np.sum(self.labels==max_i), np.sum(self.labels[(self.slices-see_range):, :, :]!=max_i), np.sum(self.labels>0), flush=True)
        # self.labels[(self.slices-see_range):, :, :] *= (self.labels[(self.slices-see_range):, :, :] == max_i).astype(np.uint32)
        #for z in range(self.slices):
        #    for x in range(512): 
        #        for y in range(512):
        #            if self.labels[z][x][y] != max_i:
        #                self.labels[z][x][y] = 0
        labels_sub = self.labels[(self.slices - see_range):, :, :]
        labels_sub = labels_sub * (labels_sub != max_i).astype(int)
        self.labels[(self.slices - see_range):, :, :] = labels_sub

        print(np.sum(self.labels>0), np.sum(self.labels[(self.slices-see_range):, :, :]!=max_i), flush=True)
