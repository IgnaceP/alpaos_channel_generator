# laplacian_solver.pyx
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:] laplacianSolver(double[:,:] H,
                                  double[:,:] K,
                                  int[:,:] boundary_i,
                                  int[:,:] boundary_j,
                                  int[:,:] outside_mask,
                                  int[:,:] chan,
                                  double resolution,
                                  double tolerance,
                                  int max_iterations):
    cdef double[:,:] result

    result = claplacianSolver(H, K, boundary_i, boundary_j, outside_mask, chan, resolution, tolerance, max_iterations)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] claplacianSolver(double[:,:] H,
                                  double[:,:] K,
                                  int[:,:] boundary_i,
                                  int[:,:] boundary_j,
                                  int[:,:] outside_mask,
                                  int[:,:] chan,
                                  double resolution,
                                  double tolerance,
                                  int max_iterations):

    cdef int i, j, rows, cols
    cdef double[:,:] lap
    cdef double l

    rows = H.shape[0]
    cols = H.shape[1]
    lap  = H

    for i in range(max_iterations):
        #laplacianIteration(H, K, resolution)
        #applyBoundaryConditions(H, boundary_i, boundary_j, outside_mask, chan)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                lap[i,j] = (H[i - 1, j] + H[i + 1, j] + H[i, j - 1] + H[i, j + 1] - (resolution ** 2) * K[i, j]) / 4.0

        h = lap

        for i in range(rows):
            for j in range(cols):
                if H[boundary_i[i, j], boundary_j[i, j]] > 0:
                    H[i, j] = H[boundary_i[i, j], boundary_j[i, j]]

                if chan[i, j] != 0:
                    H[i, j] = 0

                if outside_mask[i, j] != 0:
                    H[i, j] = 0

    return H

