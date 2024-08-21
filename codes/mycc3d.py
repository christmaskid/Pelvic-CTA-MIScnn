# Author: Yu-Tong Cheng
# Date: May 22nd, 2022

# Modified on Jul. 4th, 2022

import cc3d

class ccResult:

    def __init__(self, sample, n_slices):
        """
        Args:
            sample (array-like): 3d mask with labeling/classification in integers.
            n_slices (int): number of slices of sample. (should be same as sample.shape[0] or len(sample))
        Return:
            None
        """
        self.n_slices = n_slices
        self.original_sample = sample[:,:,:]
        self.find_cc(sample, n_slices)

    def find_cc(self, sample, n_slices):
        """
        Args: 
            Same as self.__init__()
        Return: 
            None.
        """

        print(f"{n_slices} slices.",flush=True)
        print(sample.shape)
        labels_in = sample
        # total_volume = (labels_in==1).sum()

        labels_out, N = cc3d.connected_components(
            labels_in, connectivity=26, return_N=True,
        )
        print(f"{N} components in total.",flush=True)

        vol = [0 for i in range(N+1)]
        position_list = [[] for i in range(N+1)]
        slice_list = [set() for i in range(n_slices)]

        for z in range(labels_out.shape[0]):
            for x in range(labels_out.shape[1]):
                for y in range(labels_out.shape[2]):
                    ccid = labels_out[z][x][y].item()
                    if ccid > 0:
                        position_list[ccid].append((z,x,y))
                        vol[ccid] += 1
                        slice_list[z].add(ccid)

        components_volume = []
        for i in range(1,N+1):
            if vol[i]>0:
                components_volume.append( (i, vol[i]) )

        # print(vol[1:])
        # print(position_list[1:])
        # print(slice_list)

        self.labels = labels_out
        self.N = N
        self.position_list = position_list
        self.components_volume = components_volume
        self.slice_list = slice_list

        """
        self.labels (NumPy array, 3d): Masked result of cc3d.connected components.
            Voxels of the i-th component are marked with integer i in [1..N]. 0 means background.
        self.N (int): Number of components.
        self.position_list (List( (int, int, int) )): Coordinations of voxels in component[i].
        self.components_volume (List( (int, int) )): (i, volume of component[i])
        self.slice_list (Set(int)): slice_list[z] includes all the number of components appeared in the z-th slice
                                    for z in [0..(self.n_slices-1)]
        """

    def remove_by_set(self, remove_set):
        """
        A subfunction used in self.remove_part.

        Args:
            remove_set (Set(int)): a set of numbers of the components that should be removed.
                A component is totally removed or preserved. It won't be cut in half.
        Return:
            None
        """
        # remove_volume = 0

        for i in remove_set:
            for pos in self.position_list[i]:
                self.labels[pos[0]][pos[1]][pos[2]] = 0
                # remove_volume += 1
        # print('remove_volume = ', remove_volume, flush=True)

    def remove_part(self, mode, percentage=1.0, threshold=2000):
        """
        The main function in use.

        Args:
            mode ('dust', 'not-max'):
                'dust': remove all the components with volume less than threshold in the whole sample.
                'not-max': remove all the components except the largest one in see_range.
                (see_range = percentage * self.n_slices)
            percentage (float): the percentage of sample that would be altered.
                Percentage is defined from the 'upper' side of the image (with larger number of z coordination).
            threshold (float): threshold for dusting.
        Return:
            None
        """
        
        # mode: 'dust' / 'not-max'
        # print(self.labels.shape)
        ori_shape = self.labels.shape
        n_slices = self.n_slices
        see_range = int(percentage * n_slices)
        print(f"{percentage*100}% of sample contains {see_range} slices.", flush=True)

        remove_funcs = {
            'dust':self.threshold_set(threshold=threshold), 
            'not-max':self.not_max_set(see_range=list(range((n_slices-see_range),n_slices)))
        }

        remove_set = remove_funcs[mode]()
        self.remove_by_set(remove_set)

        if self.labels.shape != ori_shape:
            print(f"shape changed: {ori_shape} -> {self.labels.shape}"); exit()
        
    def set_to_one(self):
        """
        Set all the components to binary classification.
        This should be used only after all the operations are done.
        """
        self.labels = self.original_sample * (self.labels > 0) # set all labels to 1

    def threshold_set(self, threshold=2000):
        """
        A subfunction used in self.remove_part(mode='dust').

        Args:
            threshold (float): threshold for dusting. Components with volume less than threshold
                                will be recorded in find_set.
        Return:
            find_set (Set(int)): a set of numbers of the components that should be removed.
        """

        def find_set():
            remove_set = set()
            for i, vol in self.components_volume:
                if vol < threshold:
                    remove_set.add(i)
            return remove_set

        return find_set

    def not_max_set(self, see_range):
        """
        A subfunction used in self.remove_part(mode='dust').

        Args:
            see_range (List(int)): The numbers of slices in focus.
        Return:
            find_set (Set(int)): a set of numbers of the components that should be removed.
        """

        def find_set():
            components_volume = sorted(self.components_volume, key=lambda x: x[1])
            argmax, max_volume = components_volume[-1]

            remove_set = set()
            for i in see_range:
                # print(i,self.slice_list[i], flush=True)
                remove_set = remove_set.union(self.slice_list[i])
            # print(remove_set, flush=True)
            remove_set.remove(argmax)
            # print('remove set: ', remove_set,flush=True)

            return remove_set

        return find_set
