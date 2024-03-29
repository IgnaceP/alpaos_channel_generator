import numpy as np
from scipy.interpolate import interp2d

def inject(arr, n, method='min'):
    """
        Aggregate a 2D NumPy array by reducing its size based on the reduction factor.

        Parameters:
            arr (numpy.ndarray): The input 2D NumPy array.
            n (int): The reduction factor by which the size of the array will be reduced.
                                   For example, a reduction factor of 2 reduces the size by half.

        Returns:
            numpy.ndarray: The aggregated array.
        """

    # Get the shape of the original array
    original_shape = arr.shape

    # Calculate the new shape after reduction
    new_shape = tuple(dim // n for dim in original_shape)

    # Reshape the array to have a shape divisible by the reduction factor along both axes
    arr = arr[:original_shape[0] // n * n,
          :original_shape[1] // n * n]

    # Reshape to have a shape of (n/n, n, m/n, n)
    arr_reshaped = arr.reshape(new_shape[0], n, new_shape[1], n)

    # Take the minimum along axis (1,3)
    if method == 'min':
        aggregated_arr = np.min(arr_reshaped, axis=(1, 3))
    elif method == 'max':
        aggregated_arr = np.max(arr_reshaped, axis=(1, 3))
    elif method == "mean":
        aggregated_arr = np.mean(arr_reshaped, axis=(1, 3))
    elif method == "median":
        aggregated_arr = np.median(arr_reshaped, axis=(1, 3))

    return aggregated_arr


def expand(arr, n):
    """
    Expand a 2D NumPy array by increasing its size based on the expansion factor using linear interpolation.

    Parameters:
        arr (numpy.ndarray): The input 2D NumPy array.
        n (int): The expansion factor by which the size of the array will be increased.
                           For example, an expansion factor of 2 increases the size by double.

    Returns:
        numpy.ndarray: The expanded array with linear interpolation.
    """
    # Get the shape of the original array
    original_shape = arr.shape

    # Define the coordinates of the original array
    x = np.arange(original_shape[1])
    y = np.arange(original_shape[0])

    # Create interpolation function
    f = interp2d(x, y, arr, kind='linear')

    # Generate new coordinates after expansion
    x_new = np.linspace(0, original_shape[1] - 1, original_shape[1] * n)
    y_new = np.linspace(0, original_shape[0] - 1, original_shape[0] * n)

    # Perform interpolation
    arr_expanded = f(x_new, y_new)

    return arr_expanded