import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def univariate_lr(df, feature_name, alpha=5e-5, n_cycle=500000, colors=("g", "lime")):
    y = np.array(df["Sell_price"])
    x_feature = np.array(df[feature_name])

    lr = MyLR([1, 1], alpha=alpha, n_cycle=n_cycle)
    lr.fit_(x_feature, y)
    print(f"thetas: {lr.thetas}, cost: {lr.cost_(x_feature, y)}")

    y_hat = lr.predict_(x_feature)

    # Plot
    plt.plot(x_feature, y, "o", color=colors[0], label="Sell price")
    plt.plot(x_feature, y_hat, "o",
             color=colors[1], label="Predicted sell price")

    plt.legend(loc="best")
    plt.xlabel(feature_name)
    plt.ylabel("Sell price")
    plt.show()


def multivariate_lr(df):
    x = np.array(df[["Age", "Thrust_power", "Terameters"]])
    y = np.array(df["Sell_price"])

    # lr = MyLR([1, 1, 1, 1], 9e-5, 1000000)
    lr = MyLR([380, -24, 5, -2], 9e-5, 100000)
    lr.fit_(x, y)
    print(f"thetas: {lr.thetas}, cost: {lr.cost_(x, y)}")

    y_hat = lr.predict_(x)

    # Plot
    ax1 = plt.subplot(131)
    x_age = np.array(df["Age"])
    ax1.plot(x_age, y, "o", color="b", label="Sell price")
    ax1.plot(x_age, y_hat, "o",
             color="dodgerblue", label="Predicted sell price")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Sell price")
    ax1.legend(loc="best")

    ax2 = plt.subplot(132)
    x_thrust = np.array(df["Thrust_power"])
    ax2.plot(x_thrust, y, "o", color="g", label="Sell price")
    ax2.plot(x_thrust, y_hat, "o",
             color="lime", label="Predicted sell price")
    ax2.set_xlabel("Thrust_power")
    ax2.set_ylabel("Sell price")
    ax2.legend(loc="best")

    ax3 = plt.subplot(133)
    x_dist = np.array(df["Terameters"])
    ax3.plot(x_dist, y, "o", color="darkviolet", label="Sell price")
    ax3.plot(x_dist, y_hat, "o",
             color="violet", label="Predicted sell price")
    ax3.set_xlabel("Terameters")
    ax3.set_ylabel("Sell price")
    ax3.legend(loc="best")

    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("../resources/spacecraft_data.csv")
    print("data loaded")

    # 1. Univariate
    univariate_lr(data, "Age", colors=("b", "dodgerblue"))
    univariate_lr(data, "Thrust_power", colors=("g", "lime"))
    univariate_lr(data, "Terameters", colors=("darkviolet", "violet"))

    # 2. Multiveriate
    multivariate_lr(data)
