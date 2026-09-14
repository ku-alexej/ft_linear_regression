#!/usr/bin/env python3
"""
train.py - trains a simple linear regression model (price = f(mileage))
using gradient descent, implemented from scratch (no ML/regression libraries).

Usage:
    python3 train.py [path/to/data.csv]

Saves theta0 and theta1 to theta.json so predict.py can use them.
"""

import csv
import json
import os
import sys

DATA_FILE = "data.csv"
MODEL_FILE = "theta.json"
LEARNING_RATE = 0.5
ITERATIONS = 1000


def load_data(path):
    """Read (mileage, price) pairs from a CSV file with a header row."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    mileages = []
    prices = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Dataset file is empty or has no header row.")

        # Be forgiving about column naming/case.
        fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
        if "km" not in fieldnames or "price" not in fieldnames:
            raise ValueError(
                "Dataset must have 'km' and 'price' columns "
                f"(found: {reader.fieldnames})"
            )
        km_col = fieldnames["km"]
        price_col = fieldnames["price"]

        for row_num, row in enumerate(reader, start=2):
            raw_km = (row.get(km_col) or "").strip()
            raw_price = (row.get(price_col) or "").strip()
            if raw_km == "" or raw_price == "":
                continue  # skip blank/incomplete rows
            try:
                km = float(raw_km)
                price = float(raw_price)
            except ValueError:
                raise ValueError(f"Non-numeric value on line {row_num} of {path}")
            mileages.append(km)
            prices.append(price)

    if len(mileages) == 0:
        raise ValueError(f"Dataset {path} contains no usable data rows.")

    return mileages, prices


def normalize(values):
    """Min-max scale values to [0, 1]. Returns (scaled_values, min, max)."""
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        # All values identical: avoid division by zero.
        return [0.0 for _ in values], v_min, v_max
    scaled = [(v - v_min) / (v_max - v_min) for v in values]
    return scaled, v_min, v_max


def estimate_price(theta0, theta1, mileage):
    """The hypothesis required by the subject: estimatePrice(mileage)."""
    return theta0 + (theta1 * mileage)


def train(mileages, prices, learning_rate=LEARNING_RATE, iterations=ITERATIONS):
    """
    Perform gradient descent on normalized mileage to find theta0/theta1.
    theta0 and theta1 are updated SIMULTANEOUSLY each iteration, as required.
    Returns the thetas in the *normalized* mileage space.
    """
    m = len(mileages)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        # 1. Calculate all errors using the CURRENT thetas.
        errors = [
            estimate_price(theta0, theta1, mileages[i]) - prices[i]
            for i in range(m)
        ]

        # 2. Calculate both temporary thetas from the same error set.
        tmp_theta0 = learning_rate * (sum(errors) / m)
        tmp_theta1 = learning_rate * (
            sum(errors[i] * mileages[i] for i in range(m)) / m
        )

        # 3. Update simultaneously.
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1


def denormalize_thetas(theta0_n, theta1_n, x_min, x_max):
    """
    Convert thetas trained on normalized mileage x' = (x - x_min)/(x_max - x_min)
    back into thetas that work directly on raw mileage, so predict.py can stay
    simple and use exactly the hypothesis given in the subject.
    """
    if x_max == x_min:
        # Degenerate dataset (single distinct mileage value): fall back to a
        # flat line at the trained intercept.
        return theta0_n, 0.0

    span = x_max - x_min
    theta1 = theta1_n / span
    theta0 = theta0_n - theta1 * x_min
    return theta0, theta1


def save_model(theta0, theta1, path=MODEL_FILE):
    with open(path, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=4)


def mean_squared_error(mileages, prices, theta0, theta1):
    m = len(mileages)
    squared_errors = [
        (estimate_price(theta0, theta1, mileages[i]) - prices[i]) ** 2
        for i in range(m)
    ]
    return sum(squared_errors) / m


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE

    try:
        mileages, prices = load_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    normalized_mileages, x_min, x_max = normalize(mileages)

    theta0_n, theta1_n = train(normalized_mileages, prices)
    theta0, theta1 = denormalize_thetas(theta0_n, theta1_n, x_min, x_max)

    save_model(theta0, theta1)

    mse = mean_squared_error(mileages, prices, theta0, theta1)

    print("Training complete.")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"Mean squared error on training data: {mse:.2f}")
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
