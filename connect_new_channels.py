from bwdist import bwdist
import numpy as np
from scipy.ndimage import gaussian_filter

def connectChannels(alp, region_size = 10):
    dist_newchan = bwdist(alp.newchan, return_distances = True)
    near_newchan = (dist_newchan < region_size)*alp.chan_original
    dist_chan = bwdist(alp.chan_original, return_distances = True)
    near_chan = (dist_chan < 2)*alp.newchan
    near_newchan_ij = np.column_stack(np.where(near_newchan | near_chan))

    newZ = alp.Z.copy()
    t = 0
    for i,j in near_newchan_ij:
        new_z = np.min((alp.Z*alp.newchan)[i-region_size:i+region_size, j-region_size:j+region_size])
        newZ[i,j] = min(new_z, newZ[i,j])
        t += 1
        print("Working on pixel %d/%d" % (t, len(near_newchan_ij)), end = '\r')

    Zg = gaussian_filter(newZ, 1)
    Z = np.min([Zg, newZ], axis = 0)
        
    bound = bwdist((alp.domain==0))
    Z[bound < 3] = alp.Z[bound < 3]
    
    Z[alp.domain == 0] = np.nan

    return Z