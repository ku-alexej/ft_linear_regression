#!/usr/bin/env python3
"""
bonus.py - optional visualization and precision report.

Plots the dataset, the regression line computed from theta0/theta1
(produced by our own gradient descent in train.py), and prints precision
metrics (MSE, RMSE, MAE, R^2). matplotlib is only used to DRAW the
results; the regression itself is never recomputed here.

Usage:
    python3 bonus.py [path/to/data.csv]
"""

import sys

from train import DATA_FILE, MODEL_FILE, estimate_price, load_data
from predict import load_model


def precision_report(mileages, prices, theta0, theta1):
    m = len(mileages)
    predictions = [estimate_price(theta0, theta1, x) for x in mileages]
    errors = [predictions[i] - prices[i] for i in range(m)]

    mse = sum(e ** 2 for e in errors) / m
    rmse = mse ** 0.5
    mae = sum(abs(e) for e in errors) / m

    mean_price = sum(prices) / m
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    ss_res = sum(e ** 2 for e in errors)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    print("Precision report")
    print("-----------------")
    print(f"Mean Squared Error (MSE):  {mse:.2f}")
    print(f"Root MSE (RMSE):           {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R^2 score:                 {r2:.4f}")

    return predictions


def plot(mileages, prices, theta0, theta1):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot. "
              "Install it with: pip install matplotlib --break-system-packages")
        return

    plt.scatter(mileages, prices, color="steelblue", label="Data")

    x_min, x_max = min(mileages), max(mileages)
    line_x = [x_min, x_max]
    line_y = [estimate_price(theta0, theta1, x) for x in line_x]
    plt.plot(line_x, line_y, color="firebrick", label="Regression line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("ft_linear_regression: price vs mileage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_plot.png")
    print("Plot saved to regression_plot.png")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE

    try:
        mileages, prices = load_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    theta0, theta1 = load_model(MODEL_FILE)
    if theta0 == 0.0 and theta1 == 0.0:
        print("Warning: model looks untrained (theta0=theta1=0). "
              "Run train.py first for meaningful results.")

    precision_report(mileages, prices, theta0, theta1)
    plot(mileages, prices, theta0, theta1)


if __name__ == "__main__":
    main()
