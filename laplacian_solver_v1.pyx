# laplacian_solver.pyx

cdef double[:, :] laplacianIteration(double[:, :] H,
                                      double[:, :] lap,
                                      double[:, :] K,
                                      double resolution):
    cdef int i, j, rows, cols
    cdef double n
    rows = H.shape[0]
    cols = H.shape[1]
    n = 4.

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            lap[i, j] = (H[i-1, j] + H[i+1, j] + H[i, j-1] + H[i, j+1] - (resolution**2)*K[i, j]) / n

    return lap

cdef double[:, :] applyBoundaryConditions(double[:, :] H,
                                          int[:, :] boundary_i,
                                          int[:, :] boundary_j,
                                          int[:, :] outside_mask,
                                          int[:, :] chan):

    cdef int i, j
    cdef int rows = H.shape[0]
    cdef int cols = H.shape[1]

    for i in range(rows):
        for j in range(cols):
            if H[boundary_i[i, j], boundary_j[i, j]] > 0:
                H[i, j] = H[boundary_i[i, j], boundary_j[i, j]]

    for i in range(rows):
        for j in range(cols):
            if chan[i, j] != 0:
                H[i, j] = 0

    for i in range(rows):
        for j in range(cols):
            if outside_mask[i, j] != 0:
                H[i, j] = 0

    return H

cpdef double[:,:] laplacianSolver(double[:,:] H,
                                  double[:,:] K,
                                  int[:,:] boundary_i,
                                  int[:,:] boundary_j,
                                  int[:,:] outside_mask,
                                  int[:,:] chan,
                                  double resolution,
                                  double tolerance,
                                  int max_iterations):

    cdef int i

    for i in range(max_iterations):
        H = laplacianIteration(H, H, K, resolution)
        H = applyBoundaryConditions(H, boundary_i, boundary_j, outside_mask, chan)


    return H

