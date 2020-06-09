import numpy as np


def predict_(x, theta):
    """
    Computes the prediction vector y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if theta.ndim == 2 and theta.shape[1] == 1:
        theta = theta.flatten()

    if (x.size == 0 or theta.size == 0
        or x.ndim != 2 or theta.ndim != 1
            or theta.shape[0] != x.shape[1] + 1):
        return None

    # np.dot(a,b) if a is an N-D array and b is a 1-D array
    # => it is a sum product over the last axis of a and b.
    return np.c_[np.ones(x.shape[0]), x].dot(theta)


if __name__ == "__main__":
    x = np.arange(1, 13).reshape((4, -1))

    # Example 1:
    theta1 = np.array([5, 0, 0, 0])
    print(predict_(x, theta1))
    # Ouput:
    # array([5., 5., 5., 5.])
    # Do you understand why y_hat contains only 5's here?

    # Example 2:
    theta2 = np.array([0, 1, 0, 0])
    print(predict_(x, theta2))
    # Output:
    # array([1., 4., 7., 10.])
    # Do you understand why y_hat == x[:,0] here?

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
    print(predict_(x, theta3))
    # Output:
    # array([9.64, 24.28, 38.92, 53.56])

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5])
    print(predict_(x, theta4))
    # Output:
    # array([12.5, 32., 51.5, 71.])
