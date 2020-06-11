import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def add_polynomial_features(x, power):
    if x.ndim == 1:
        x = x[:, np.newaxis]

    new_x = x
    for _ in range(power - 1):
        new_x = np.c_[new_x, new_x[:, -1] * new_x[:, 0]]

    return new_x


def get_poly_cost(x, y, poly, alpha=0.001, n_cycle=1000, thetas=None):
    if thetas is None:
        thetas = [1] * (poly + 1)

    lr = MyLR(thetas, alpha=alpha, n_cycle=n_cycle)
    poly_x = add_polynomial_features(x, poly)

    lr.fit_(poly_x, y)

    cost = lr.cost_(poly_x, y)
    print(f"Poly {poly}: {cost} | {repr(lr.thetas)}")

    return cost


def main():
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")

    x = np.array(data["Micrograms"])
    y = np.array(data["Score"])

    cost_lst = []

    cost_lst.append(
        get_poly_cost(x, y, 2, alpha=9e-4, n_cycle=1000,
                      thetas=[91.9896726, -10.88120698,   0.23854169]))
    cost_lst.append(
        get_poly_cost(x, y, 3, alpha=8e-5, n_cycle=1000,
                      thetas=[83.89404375, -1.22799024, -2.79045065,  0.27281495]))
    cost_lst.append(
        get_poly_cost(x, y, 4, alpha=2e-6, n_cycle=1000,
                      thetas=[60.9152876,  35.10744969, -20.44068329, 3.63034827, -0.21922759]))
    cost_lst.append(
        get_poly_cost(x, y, 5, alpha=5e-8, n_cycle=1000,
                      thetas=[37.47107575,  32.8691159,  14.7536331, -16.00899375, 3.56106907,  -0.24224376]))

    # No enough training for the rest
    # (it's too loooong, I don't have the time)
    # cost_lst.append(
    #     get_poly_cost(x, y, 6, alpha=1e-9, n_cycle=100000, thetas=[
    #         1.02679125,  1.03684944,  1.04412233,  0.99991945,  0.69793448, -0.30702394,  0.02687771]))
    # cost_lst.append(
    #     get_poly_cost(x, y, 7, alpha=4e-11, n_cycle=100000, thetas=[
    #         1.00966418,  1.00238758,  0.96892669,  0.84963847,  0.51005825, -0.11575169, -0.01859237,  0.00325301]))
    # cost_lst.append(
    #     get_poly_cost(x, y, 8, alpha=1e-12, n_cycle=10000, thetas=[
    #         0.99976379,  0.99544007,  0.97963496,  0.9258643,  0.75994343, 0.3338223, -0.35310088,  0.0722765, -0.00449107]))
    # cost_lst.append(
    #     get_poly_cost(x, y, 9, alpha=2e-14, n_cycle=10000, thetas=[
    #         0.99955827,  0.99833447,  0.99388472,  0.97805356,  0.9243213, 0.75783392,  0.32811412, -0.36667952,  0.07607999, -0.00473927]))
    # get_poly_cost(x, y, 10, alpha=5e-5, n_cycle=10000)

    # Ploting all cost
    plt.bar(list(range(2, len(cost_lst) + 2)), cost_lst)
    plt.show()


if __name__ == "__main__":
    main()
