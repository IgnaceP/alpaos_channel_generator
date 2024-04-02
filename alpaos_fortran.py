import numpy as np
from numba import *
from bwdist import bwdist
from pysheds.grid import Grid
from pysheds.view import Raster
import time
import cython
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import binary_dilation, binary_erosion

import alpaos_channel_generator.laplacian_fortran as lapl_f
from alpaos_channel_generator.inject_interpolate import inject, expand

class AlpaosChannelCreator:
    def __init__(self, Hstart, K, chan, domain, H0, Z, resolution, fn_tif = 'test.tif'):
        self.Hstart             = np.asarray(Hstart, dtype = "float64")
        self.H              = Hstart.copy()
        self.K              = K
        self.chan           = chan
        self.domain         = domain
        self.H0             = H0
        self.Z              = Z
        self.Z_orig         = Z.copy()
        self.fn_tif         = fn_tif
        self.newchan        = np.zeros_like(self.chan)
        self.resolution     = resolution

        self.H_maxs         = []
        self.total_counter  = 0

        self.chan_original  = self.chan.copy()
        
        self.outsideNeighbours()
        self.boundary_i, self.boundary_j = self.initiateBoundaryConditions(self.domain)
        self.initiateGrid()
        self.findBoundary(w = 10)

        self.H              = np.asfortranarray(self.H, dtype=np.float64)
        self.K              = np.asfortranarray(self.K, dtype=np.float64)
        self.boundary_i     = np.asfortranarray(self.boundary_i, dtype=int)
        self.boundary_j     = np.asfortranarray(self.boundary_j, dtype=int)
        self.chan           = np.asfortranarray(self.chan, dtype=int)
        self.outside_mask   = np.asfortranarray(self.outside_mask, dtype=int)

    def save(self, pathname):
        """
        Method to save channel generator as a pickle
        :param pathname: directory and filename to save as
        :return:
        """

        with open(pathname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(pathname):
        """
        Method to load channel generator from a pickle
        :param pathname: directory and filename to load
        :return:
        """

        with open(pathname, 'rb') as input:
            alp = pickle.load(input)

        return alp
    def outsideNeighbours(self):
        """
        Function to calculate the number of cardinal neigbours which are outside the domain
        """
        self.outside_mask = (self.domain == 0)
        outside_neighbours = np.zeros_like(self.domain)
        outside_neighbours[1:-1, 1:-1] = np.sum([
            self.outside_mask[:-2, 1:-1] , self.outside_mask[2:, 1:-1] ,
            self.outside_mask[1:-1, :-2] , self.outside_mask[1:-1, 2:] ], axis = 0)

        self.outside_neighbours = np.asarray(outside_neighbours, dtype = "int")
        self.outside_neighbours[self.outside_neighbours > 3] = 3

    @staticmethod
    def initiateBoundaryConditions(domain):
        """
        Function to initiate the boundary conditions
        :return:
        """
        outside_mask = (domain == 0)
        outside_mask_dist = bwdist(outside_mask)

        boundary = ((outside_mask_dist < 2) & (outside_mask_dist > 0))
        innerboundary = (outside_mask_dist < 3) & (outside_mask_dist > 1.5)

        innerboundary_dist_ij = bwdist(innerboundary, return_indices=True, return_distances=False)
        boundary_i = innerboundary_dist_ij[0] * boundary
        boundary_j = innerboundary_dist_ij[1] * boundary

        return boundary_i, boundary_j

    def initiateGrid(self):
        """
        Function to initiate the grid
        :return:
        :return:
        """

        grid    = Grid.from_raster(self.fn_tif)
        grid.read_raster(self.fn_tif)

        self.grid = grid

    @staticmethod
    @jit(nopython=True)
    def slope(H, outside_neighbours, res = 1):
        """
        Function to calculate a slope map from a water surface map
        not using np.gradient
        :param H:
        :return:
        """
        dH = np.zeros_like(H)
        dHdx = (H[2:, 1:-1] - H[:-2, 1:-1]) / (2 * res)
        dHdy = (H[1:-1, 2:] - H[1:-1, :-2]) / (2 * res)
        dH[1:-1, 1:-1] = (dHdx ** 2 + dHdy ** 2) ** 0.5
        dH *= (outside_neighbours == 0)

        return dH

    def findBoundary(self, w = 5):
        self.boundary = (bwdist(~np.asarray(self.domain, dtype = bool)) < w)
    @staticmethod
    @jit(nopython=True)
    def findNeighbours(chan):
        """
        Function to find the neighbours of a channel
        :param chan:
        :return:
        """
        neighbours = np.zeros_like(chan)
        neighbours[1:-1, 1:-1] = chan[:-2, 1:-1] + chan[2:, 1:-1] + chan[1:-1, :-2] + chan[1:-1, 2:]

        neighbours *= (chan == 0)
        neighbours = (neighbours > 0)
        return neighbours

    def flowdir(self):
        Hgrid = Raster(self.H, self.grid.viewfinder)
        self.fdir = self.grid.flowdir(Hgrid)
    def watershed(self, index):
        """
        Function to apply the watershed algorithm to a water surface map
        :param H:
        :param grid:
        :param indices:
        :return:
        """
        rows, cols = self.H.shape
        i = index[0]
        j = index[1]
        catch = self.grid.catchment(x=j, y=rows - i, fdir=self.fdir)

        return catch

    def calculate_ES(self, index):
        """
        Function to calculate the energy E(S) which is represented by the average value of H over the unchanneled portions of the tidal basin
        :param H:
        :param res:
        :return:
        """

        ws = self.watershed(index)
        Es = np.mean(self.H[ws == 1])

        return Es

    def calculate_PS(self, i,j, T = "Hmean"):
        """
        Function to calculate the probability P(S)
        :param indices:
        :return:
        """

        if T == "Hmean":
            T = np.nanmean(self.H)


        es = self.calculate_ES([i, j])
        ps = np.exp(-es / T)

        return ps

    def agg(self, agg_level=2, stepsize = 1000, tolerance = 1e-6):
        """
        Method to decrease resolution and solve the laplacian equation
        :param agg_level: 
        :param iterations: 
        :return: 
        """

        if not agg_level in [1,2,4,8,16,32,64]:
            raise ValueError('Aggregation level should be a power of 2.')

        if self.H.shape[0] % agg_level != 0 or self.H.shape[1] % agg_level != 0:
            raise ValueError('Dimensions of input rasters should be divisible by the aggregation level.')

        print('Agreggating over level %i ...' % agg_level, end = '\r')

        t0 = time.time()
        H_agg = inject(self.H, agg_level)
        K_agg = inject(self.K, agg_level)

        domain_agg = inject(self.domain, agg_level, method='max')
        if np.log2(agg_level)-1 > 0:
            domain_agg = binary_dilation(domain_agg, iterations=int(np.log2(agg_level)))
        domain_agg[0, :] = 0;
        domain_agg[:, -1] = 0;
        domain_agg[-1, :] = 0;
        domain_agg[:, 0] = 0
        bi_agg, bj_agg = self.initiateBoundaryConditions(domain_agg)
        om_agg = np.logical_not(domain_agg)
        ch_2 = inject(self.chan, agg_level, method = 'max')
        rows_agg, cols_agg = H_agg.shape
        res_agg = self.resolution * agg_level


        # ------------------------------------------------------------------------------------------------------------ #
        # Solve the laplacian equation

        H_agg_prev = np.zeros_like(H_agg)

        t = 0
        self.H_agg_box = H_agg.copy()
        while np.quantile(np.abs(-H_agg + H_agg_prev),.99) > tolerance:

            H_agg_prev = H_agg.copy()

            # calculate water surface at aggregated resolution
            lapl_f.laplaciansolver(h=H_agg, k=K_agg, boundary_i=bi_agg, boundary_j=bj_agg,
                                   outside_mask=om_agg, chan=ch_2, rows=rows_agg, cols=cols_agg,
                                   iterations=stepsize, resolution=res_agg)

            t += stepsize
            self.H_agg_box = np.dstack([self.H_agg_box, H_agg])

        self.H_agg_prev = H_agg_prev
        self.H_agg = H_agg
        self.diff_agg = np.quantile(np.abs(-H_agg + H_agg_prev),.99)
        # ------------------------------------------------------------------------------------------------------------ #

        # expand to original resolution and set to alpaos object
        self.H = np.asfortranarray(expand(H_agg, agg_level))
        self.H[self.H[self.boundary_i, self.boundary_j] > 0] = self.H[self.boundary_i, self.boundary_j][
            self.H[self.boundary_i, self.boundary_j] > 0]

        t1 = time.time()
        print('Agreggating over level %i took %i iterations over %.0f seconds.' % (agg_level, t, (t1-t0)))

    def laplacianSolver(self, H_max_iterations=100000, min_iterations=500, tolerance=1e-6, stepsize = 10000):

        """
        Function to solve the laplacian equation
        """

        self.H_prev             = self.H.copy()
        continue_flag           = True
        self.counter            = 0

        while continue_flag:
            # calculate water surface
            lapl_f.laplaciansolver(h=self.H, k=self.K, boundary_i=self.boundary_i, boundary_j=self.boundary_j,
                                   outside_mask=self.outside_mask, chan=self.chan, rows=self.H.shape[0],
                                   cols=self.H.shape[1], iterations=stepsize, resolution=self.resolution)

            # evaluate convergence
            self.H_maxs.append(np.nanmax(self.H))

            self.counter += stepsize
            self.total_counter += stepsize

            if self.counter >= H_max_iterations:
                continue_flag = False
                print("Maximum number of iterations reached.")
            elif self.counter >= min_iterations:
                diff = np.quantile(-self.H + self.H_prev,.99)
                if diff < tolerance:
                    continue_flag = False
                else:
                    self.H_prev = self.H.copy()

    def solve(self, iterations = 100, gamma = 9810, tau_crit = 0.001,
              H_max_iterations = 100000, H_min_iterations = [100,10], tolerance = 1e-7, T = "Hmean",
              stepsize = 1000, agg_levels = [8,4,2]):

            self.tau_crit                       = tau_crit
            self.catch                          = np.zeros_like(self.chan)

            for iter in range(iterations):

                print(f"Step: {iter+1}/{iterations}")

                if agg_levels:
                    for agg_level in agg_levels:
                        self.agg(agg_level = agg_level, stepsize = 100, tolerance = tolerance*stepsize)

                print("Solving Poisson equation ...", end='\r')
                time0 = time.time()
                self.laplacianSolver(H_max_iterations = H_max_iterations, min_iterations = H_min_iterations[min(self.total_counter,1)],
                                     tolerance = tolerance*stepsize, stepsize = stepsize)
                time1 = time.time()
                print(f'Solving Poisson equation took {self.counter} iterations over {int(time1-time0)} seconds.')

                if np.sum(np.isnan(self.H)) > 0:
                    raise ValueError('There are NaN values in the water surface map.')

                # calculate shear stress
                self.Hslope                 = self.slope(self.H, self.outside_neighbours, res = self.resolution)
                self.tau = tau              = ~self.boundary * gamma * (self.H0 + self.H - self.Z) * self.Hslope

                # get neighbouring cells of channels
                self.neigh                  = self.findNeighbours(self.chan)

                # calculat flow direction
                self.flowdir()

                # select sites with expected activity
                tau_excess                                      = tau - tau_crit
                self.exceedance = exceedance                    = tau_excess * self.neigh
                exceedance_i, exceedance_j                      = np.where(exceedance > 0)
                exceedance_flat                                 = exceedance[exceedance_i, exceedance_j]
                exceedance_ij_sorted                            = np.argsort(exceedance_flat)[::-1]
                n_exceedances                                   = len(exceedance_ij_sorted)

                # select sites where the creeks will expand
                t = 0
                new_n = 0
                while n_exceedances > 0:
                    R                       = np.random.rand()
                    i,j = exceedance_i[exceedance_ij_sorted[t]], exceedance_j[exceedance_ij_sorted[t]]
                    Ps = self.calculate_PS(i, j, T = T)
                    if R > Ps:
                        new_chan = True
                    elif t == n_exceedances-1:
                        rt = np.random.randint(0, n_exceedances)
                        i, j = exceedance_i[rt],exceedance_j[rt]
                        new_chan = True
                    else:
                        t += 1
                        new_chan = False

                    if new_chan:
                        print(f'Took the the {t+1}th of {n_exceedances} candidates: i = {i}, j = {j}')

                        # set new channel cell
                        self.chan[i,j] = 1
                        self.newchan[i,j] = 1

                        # calculate watershed for new channel cell
                        catch = float(np.sum(self.watershed([i, j]))) * self.resolution ** 2
                        # based on catchment area, set channel dimensions
                        area = 10 ** (-1.38) * (catch) ** 0.58
                        B = (area * .33 ** -1) ** .5
                        D = np.min([area / B, 10])
                        self.Z[i,j] = self.Z_orig[i,j] - D

                        # in one iterations, only distant channels can expand
                        dist_ij = (exceedance_ij_sorted-i)**2 + (exceedance_ij_sorted-j)**2
                        distant_channels_mask = (dist_ij > 10)

                        if np.sum(distant_channels_mask) > 0:
                            exceedance_ij_sorted = exceedance_ij_sorted[distant_channels_mask]
                            t += 1
                            if t == len(exceedance_ij_sorted):
                                break
                        else:
                            break

                        # update counters
                        new_n += 1

                        new_chan = False

                        if new_n == 3:
                            break

                # calculate watershed for all new channel cells
                new_chan_i, new_chan_j = np.where(self.newchan == 1)

                for counter, (i, j) in enumerate(zip(new_chan_i, new_chan_j)):
                    catch           = float(np.sum(self.watershed([i,j])))*self.resolution**2
                    self.catch[i,j] = catch
                    area            = 10**(-1.38)*(catch)**0.58
                    B               = (area*.33**-1)**.5
                    D               = np.min([area/B,10])
                    Bi              = int(B//self.resolution+1)//2+1
                    for bi in range(Bi):
                        self.chan[i-bi:i+bi, j-bi:j+bi] = 1
                        self.Z[i-bi:i+bi, j-bi:j+bi] = self.Z_orig[i-bi:i+bi, j-bi:j+bi] - D
                    print('Recalculating channel dimensions: %.2f %%' % (100*(counter+1)/len(new_chan_i)), end = '\r')
                print('\n')


                print("---------------------------------------------------------------------------------------------")

    def plot(self):

        # Plot the final grid
        tau = self.tau
        tau[self.domain == 0] = np.nan

        fig = plt.figure(figsize=(12, 8))
        ax = []
        ax.append(fig.add_subplot(3, 2, 1))
        ax.append(fig.add_subplot(3, 2, 3))
        ax.append(fig.add_subplot(3, 2, 5))
        ax.append(fig.add_subplot(3, 2, (2, 4)))
        ax.append(fig.add_subplot(3, 2, 6))

        imH     = ax[0].imshow(self.H, cmap='viridis')
        contH   = ax[0].contour(self.H, levels=10, colors='white', linewidths=0.25, alpha = .5)
        imTau   = ax[1].imshow(np.abs(self.tau * self.domain), cmap='jet', vmax = 2*self.tau_crit)
        imZ     = ax[2].imshow(self.Z, cmap='gist_earth')
        imChan  = ax[3].imshow(
            (self.domain * (1 + .25 * self.boundary + 1 * self.chan_original + 10 * (self.chan - self.chan_original))),
            cmap='Reds')
        lHmax = ax[4].plot(self.H_maxs)

        ax[4].set_ylabel('Hmax [m]')
        ax[4].set_xlabel('Iterations')

        for i, im in enumerate([imH, imTau, imZ, imChan]):
            ax[i].set_title([r'$H_1$ [cm]', 'Shear stress [Pa]', "Bottom elevation [z]", 'Channel network'][i])

            if i != 3:
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = fig.colorbar(im, cax=cax, orientation='vertical')

        return fig, ax