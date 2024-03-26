import numpy as np
from numba import *
from bwdist import bwdist

#@jit(nopython=True)
def laplacianIteration(H, K, outside_neighbours):
    laplacian = np.zeros_like(H)
    laplacian[1:-1, 1:-1] = ((H[:-2, 1:-1] + H[2:, 1:-1] + H[1:-1, :-2] + H[1:-1, 2:] + K[1:-1, 1:-1])
                             / (4 - outside_neighbours[1:-1, 1:-1]))
    return laplacian

#@jit(nopython=True)
def applyBoundaryConditions(H, boundary_i, boundary_j, outside_mask, chan):
    newH = H.copy()

    # assumption 1: no flux perpendicular to the boundary
    # method 1: avoid multidimensional indexing
    bif = boundary_i.flatten()
    bjf = boundary_j.flatten()
    linear_indices = bif * H.shape[1] + bjf
    border = np.take(H, linear_indices).reshape(H.shape)
    newH *= (border <= 0)
    newH += border*(border > 0)
    # method 2: with multidimensional indexing
    #border = H[boundary_i, boundary_j]
    #newH[border > 0] = border[border > 0]

    # assumption 2: local deviation of water surface is zero in the channels
    newH *= (chan == 0)

    # no values outside the domain
    newH *= ~outside_mask

    return newH, border

#@jit(nopython=True)
def laplacianSolver(H, K, outside_neighbours, boundary_i, boundary_j, outside_mask, chan, iterations=100):
    maxs = np.zeros(iterations)
    #stds = np.zeros(iterations)

    for i in range(iterations):
        # calculate water surface
        H = laplacianIteration(H, K, outside_neighbours)
        H,border = applyBoundaryConditions(H, boundary_i, boundary_j, outside_mask, chan)

        # calculate shear stress
        tau = 0

        # evaluate convergence
        maxs[i] = np.nanmax(H)
        #stds.append(np.nanstd(H))

        # print progress
        # print(f'Progress {i+1}/{iterations}', end = '\r')
    return H, maxs, border
class AlpaosChannelCreator:
    def __init__(self, Hstart, K, chan, domain):
        self.Hstart = Hstart
        self.H = Hstart.copy()
        self.K = K
        self.chan = chan
        self.domain = domain

        self.outsideNeighbours()
        self.initiateBoundaryConditions()

    def outsideNeighbours(self):
        """
        Function to calculate the number of cardinal neigbours which are outside the domain
        """
        self.outside_mask = (self.domain == 0)
        outside_neighbours = np.zeros_like(self.domain)
        outside_neighbours[1:-1, 1:-1] = np.sum([
            self.outside_mask[:-2, 1:-1] , self.outside_mask[2:, 1:-1] ,
            self.outside_mask[1:-1, :-2] , self.outside_mask[1:-1, 2:] ], axis = 0)

        self.outside_neighbours = outside_neighbours

    def initiateBoundaryConditions(self):
        """
        Function to initiate the boundary conditions
        :return:
        """
        outside_mask = (self.domain == 0)
        outside_mask_dist = bwdist(outside_mask)

        boundary = ((outside_mask_dist < 2) & (outside_mask_dist > 0))
        innerboundary = (outside_mask_dist < 3) & (outside_mask_dist > 1.5)

        innerboundary_dist_ij = bwdist(innerboundary, return_indices=True, return_distances=False)
        self.boundary_i = innerboundary_dist_ij[0] * boundary
        self.boundary_j = innerboundary_dist_ij[1] * boundary

    def solve(self, iterations = 100):
        return laplacianSolver(H = self.H, K = self.K, outside_neighbours = self.outside_neighbours,
                               boundary_i = self.boundary_i, boundary_j = self.boundary_j,
                               outside_mask = self.outside_mask, chan = self.chan,
                               iterations=iterations)