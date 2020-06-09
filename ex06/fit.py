import numpy as np


# Just here to test result
def predict_(x, theta):
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


def fit_(x, y, theta, alpha, n_cycles):
    """
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of training examples, number of features).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        n_cycles: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
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
    new_theta = theta.astype("float64")
    for _ in range(n_cycles):
        nabla = x_padded.T.dot(x_padded.dot(new_theta) - y) / y.shape[0]
        new_theta = new_theta - alpha * nabla

    return new_theta


if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                  [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, 0.0005, 42000)
    print(theta2)
    # Output:
    # array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    # Example 1:
    print(predict_(x, theta2))
    # Output:
    # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]]
