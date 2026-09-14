#!/usr/bin/env python3
"""
predict.py - asks the user for a mileage and prints the estimated price,
using estimatePrice(mileage) = theta0 + (theta1 * mileage).

Usage:
    python3 predict.py

If theta.json does not exist yet (training hasn't been run), theta0 and
theta1 default to 0, as required by the subject.
"""

import json
import os
import sys

MODEL_FILE = "theta.json"


def load_model(path=MODEL_FILE):
    """Load theta0/theta1. Defaults to (0, 0) if no model has been trained."""
    if not os.path.isfile(path):
        return 0.0, 0.0

    try:
        with open(path) as f:
            data = json.load(f)
        return float(data["theta0"]), float(data["theta1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"Warning: {path} is corrupted or invalid. Using theta0=0, theta1=0.")
        return 0.0, 0.0


def estimate_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


def read_mileage(prompt="Enter a mileage (km): "):
    """
    Prompt for a mileage and validate it.
    Returns a float, or None if the user wants to quit.
    Raises ValueError with a friendly message for bad input.
    """
    raw = input(prompt).strip()

    if raw == "":
        raise ValueError("Mileage cannot be empty.")

    if raw.lower() in ("q", "quit", "exit"):
        return None

    try:
        mileage = float(raw)
    except ValueError:
        raise ValueError(f"'{raw}' is not a valid number.")

    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")

    return mileage


def main():
    theta0, theta1 = load_model()

    while True:
        try:
            mileage = read_mileage()
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

        if mileage is None:
            print("Goodbye.")
            break

        price = estimate_price(theta0, theta1, mileage)
        price = max(price, 0.0)  # a car's price cannot be negative
        print(f"Estimated price for {mileage:.0f} km: {price:.2f}")
        break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        sys.exit(0)
