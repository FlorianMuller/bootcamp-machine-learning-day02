import numpy as np


def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1,
            containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    if theta.ndim == 2 and theta.shape[1] == 1:
        theta = theta.flatten()

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.ndim != 2 or y.ndim != 1 or theta.ndim != 1
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]):
        return None

    x_padded = np.c_[np.ones(x.shape[0]), x]

    return x_padded.T.dot(x_padded.dot(theta) - y) / y.shape[0]


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    # x = np.array([
    #     [-7, -9],
    #     [-2, 14],
    #     [14, -1],
    #     [-4, 6],
    #     [-9, 6],
    #     [-5, 11],
    #     [-11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([0, 3, 0.5, -6])

    # Example :
    print(gradient(x, y, theta1))
    # Output:
    # [ -33.71428571  -37.35714286  183.14285714 -393. ]
    # me:      array([ -33.71428571,  -37.35714286, 183.14285714, -393.])
    # subject: array([                -37.35714286, 183.14285714, -393.])
    # (The subject was missing one theta)

    # Example :
    theta2 = np.array([0, 0, 0, 0])
    print(gradient(x, y, theta2))
    # Output:
    # me:      array([ -0.71428571,   0.85714286, 23.28571429, -26.42857143])
    # subject: array([                0.85714286, 23.28571429, -26.42857143])
    # (The subject was missing one theta)
