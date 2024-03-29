# laplacian_solver.pyx
import numpy as np
cimport numpy as np

def laplacianIteration(np.ndarray[np.float64_t, ndim=2] H,
                        np.ndarray[np.float64_t, ndim=2] K,
                        np.float64_t resolution):
    cdef np.ndarray[np.float64_t, ndim=2] laplacian = np.zeros_like(H)
    cdef Py_ssize_t i, j
    cdef np.float64_t n

    for i in range(1, H.shape[0]-1):
        for j in range(1, H.shape[1]-1):
            laplacian[i, j] = (H[i-1, j] + H[i+1, j] + H[i, j-1] + H[i, j+1] - (resolution**2)*K[i, j]) / (4.0)

    return laplacian.astype(np.float64)

def applyBoundaryConditions(np.ndarray[np.float64_t, ndim=2] H,
                            np.ndarray[np.int32_t, ndim=2] boundary_i,
                            np.ndarray[np.int32_t, ndim=2] boundary_j,
                            np.ndarray[np.int32_t, ndim=2] outside_mask,
                            np.ndarray[np.int32_t, ndim=2] chan):

    cdef np.ndarray[np.float64_t, ndim=2] newH = np.copy(H)

    cdef int i, j
    for i in range(boundary_i.shape[0]):
        for j in range(boundary_i.shape[1]):
            if H[boundary_i[i, j], boundary_j[i, j]] > 0:
                newH[i, j] = H[boundary_i[i, j], boundary_j[i, j]]

    for i in range(newH.shape[0]):
        for j in range(newH.shape[1]):
            if chan[i, j] != 0:
                newH[i, j] = 0

    for i in range(newH.shape[0]):
        for j in range(newH.shape[1]):
            if outside_mask[i, j] != 0:
                newH[i, j] = 0

    return newH

def laplacianSolver(np.ndarray[np.float64_t, ndim=2] H,
                    np.ndarray[np.float64_t, ndim=2] K,
                    np.ndarray[np.int32_t, ndim=2] boundary_i,
                    np.ndarray[np.int32_t, ndim=2] boundary_j,
                    np.ndarray[np.int32_t, ndim=2] outside_mask,
                    np.ndarray[np.int32_t, ndim=2] chan,
                    np.float64_t resolution,
                    np.float64_t tolerance,
                    int max_iterations):

    cdef np.ndarray[np.float64_t, ndim=2] oldH = np.copy(H)
    cdef np.int32_t continue_flag = 1
    cdef np.int32_t counter = 0
    cdef np.ndarray[np.float64_t, ndim=1] H_maxs = np.zeros(max_iterations)

    for _ in range(max_iterations):
        H = laplacianIteration(H, K, resolution)
        H = applyBoundaryConditions(H, boundary_i, boundary_j, outside_mask, chan)

        H_maxs[counter] = np.nanmax(H)
        counter += 1

        if (np.nanmax(((H - oldH) ** 2) ** .5)) < tolerance:
            break
        elif counter >= max_iterations:
            break

    H_maxs = H_maxs[:counter]

    return H, H_maxs, counter
