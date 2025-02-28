import numpy as np
from numba import *
from bwdist import bwdist
from pysheds.grid import Grid
from pysheds.view import Raster
import alpaos_channel_generator.laplacian_solver as laplace_cython
import time
import cython
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        self.H                  = np.asarray(self.H, dtype="float64")
        self.K                  = np.asarray(self.K, dtype="float64")
        self.outside_neighbours = np.asarray(self.outside_neighbours, dtype="int")
        self.chan               = self.chan.astype(np.int32)
        self.boundary_i         = self.boundary_i.astype(np.int32)
        self.boundary_j         = self.boundary_j.astype(np.int32)
        self.outside_mask       = self.outside_mask.astype(np.int32)
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
    def laplacianSolver(self, H_max_iterations=100000, min_iterations=500, tolerance=1e-7, stepsize = 10000):

        """
        Function to solve the laplacian equation
        """

        self.H_prev             = np.zeros_like(self.H)

        continue_flag           = True
        self.counter            = 0

        while continue_flag:
            # calculate water surface
            #self.H = laplace_cython.laplacianIteration(self.H, self.K, self.resolution)
            #self.H = laplace_cython.applyBoundaryConditions(self.H, self.boundary_i, self.boundary_j,
            #                                                self.outside_mask, self.chan)
            self.H = laplace_cython.laplacianSolver(
                          self.H,
                                self.K,
                                self.boundary_i,
                                self.boundary_j,
                                self.outside_mask,
                                self.chan,
                                self.resolution,
                                tolerance,
                                stepsize)

            self.H = np.asarray(self.H, dtype="float64")

            # evaluate convergence
            self.H_maxs.append(np.nanmax(self.H))

            self.counter += stepsize
            self.total_counter += stepsize

            if self.counter >= H_max_iterations:
                continue_flag = False
                print("Maximum number of iterations reached.")
            elif self.counter >= min_iterations:
                diff = np.nanmax(((self.H - self.H_prev) ** 2) ** .5)
                print(
                    f'Step {self.counter // stepsize} | iteration {self.counter} | diff: {np.round(diff, 4)}',end = '\r')
                if diff < tolerance:
                    continue_flag = False
                else:
                    self.H_prev = self.H.copy()



    def solve(self, iterations = 100, gamma = 9810, tau_crit = 0.001,
              H_max_iterations = 100000, H_min_iterations = [100,10], tolerance = 1e-7, T = "Hmean",
              stepsize = 10000):

            print('Solving initial Poisson equation for H.')

            self.catch                          = np.zeros_like(self.chan)

            for iter in range(iterations):
                time0                       = time.time()
                self.laplacianSolver(H_max_iterations = H_max_iterations, min_iterations = H_min_iterations[min(self.total_counter,1)],
                                     tolerance = tolerance*stepsize, stepsize = stepsize)
                time1                       = time.time()

                print(f"Step: {iter+1}/{iterations}")
                if time1 - time0 > 10:
                    print(f'Solving Poisson equation for H took {np.round((time1-time0)/60,2)} minutes to run {self.counter} iterations.')

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

                t = 0
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
                        print(f'took the the {t+1}th of {n_exceedances} candidates.')

                        self.chan[i,j] = 1
                        self.newchan[i,j] = 1
                        catch = float(np.sum(self.watershed([i, j]))) * self.resolution ** 2
                        area = 10 ** (-1.38) * (catch) ** 0.58
                        B = (area * .33 ** -1) ** .5
                        D = np.min([area / B, 10])
                        self.Z[i,j] = self.Z_orig[i,j] - D
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
                    print(' Recalculating channel dimensions: %.2f %%' % (100*(counter+1)/len(new_chan_i)), end = '\r')
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
        imTau   = ax[1].imshow(np.abs(self.tau * self.domain), cmap='jet')
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