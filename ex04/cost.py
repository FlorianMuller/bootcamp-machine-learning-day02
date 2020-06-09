import numpy as np


def cost_(y, y_hat):
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    if y_hat.ndim == 2 and y_hat.shape[1] == 1:
        y_hat = y_hat.flatten()

    if (y.size == 0 or y_hat.size == 0
        or y.ndim != 1 or y_hat.ndim != 1
            or y.shape != y_hat.shape):
        return None

    y_diff = y_hat - y
    return np.dot(y_diff, y_diff) / (2 * y.shape[0])


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Example 1:
    print(cost_(X, Y))
    # Output:
    # 2.142857142857143
    # (AND NOT 4.285714285714286 like it's written in the subject)

    # Example 2:
    print(cost_(X, X))
    # Output:
    # 0.0
