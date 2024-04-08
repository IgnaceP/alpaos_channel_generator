import numpy as np
from bwdist import bwdist

def calculatePathLength(domain, pl_start, res, print_flag = True):
    """
    Calculate the path length of a channel network
    :param domain: numpy array (m x n) with the domain as ones and zeros for cells outside the domain
    :param pl_start: numpy array (m x n) with the starting situation for the path length calculation
    :param res: float, resolution of the domain
    :param print_flag: boolean, print the maximum distance
    :return: numpy array (m x n) with the path length of the channel network
    """
    pl = pl_start.copy()

    counter = 0
    while True:
        dists, [ni, nj] = bwdist((pl > 0), return_distances=True, return_indices=True)
        neigh = (dists == 1)
        ni *= neigh; nj *= neigh
        new = (pl[ni,nj]+res*neigh)*domain
        pl[new > 0] = new[new > 0]

        counter += 1

        if print_flag:
            print("Max distance: ", counter*res, " m", end = '\r')

        if np.sum(new) == 0:
            break

        if counter > 1e6:
            break

    if print_flag:
        print("Max distance: ", counter * res, " m",)

    return pl

