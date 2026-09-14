# ft_linear_regression

A from-scratch implementation of simple linear regression (one feature)
trained with gradient descent, used here to predict a car's price from
its mileage.

This project restricted the use of external libraries to handle the core
implementation. `matplotlib` was used exclusively to build the plot for
the bonus part.

## Purpose

- `train.py` reads `data.csv` and learns `theta0`/`theta1` for the hypothesis
  `estimatePrice(mileage) = theta0 + (theta1 * mileage)` using gradient descent
  implemented by hand (no library does the regression for us).
- `predict.py` loads the trained thetas and prints an estimated price for
  a mileage you enter.
- `bonus.py` (optional) plots the data and regression line and prints
  precision metrics (MSE, RMSE, MAE, R²).

## Requirements

- Python 3
- Standard library only for the mandatory part (`csv`, `json`, `os`, `sys`)
- `matplotlib` only for the bonus plot

## Installation

```bash
# for bonus part
pip install matplotlib --break-system-packages
```

# Training the model

```bash
# uses data.csv by default
python3 train.py

# uses other data
python3 train.py path/to/other_dataset.csv
```

This prints the learned `theta0`/`theta1` and the training MSE, and
saves them to `theta.json`:

```json
{
    "theta0": 8499.599649893675,
    "theta1": -0.021448963591326264
}
```

Internally, mileage is min-max normalized before gradient descent, and 
the resulting thetas are converted back ("denormalized") so they work
directly on raw mileage. `predict.py` therefore never needs to know
normalization happened.

## Running a prediction

```bash
python3 predict.py
```

You'll be prompted for a mileage and shown the estimated price. Before
any training has been run, `theta0` and `theta1` default to `0`, so every
prediction is `0` until you train.

Invalid input (empty, non-numeric, or negative mileage) is rejected
with a clear message instead of crashing; enter `q` to quit.

## Bonus

```bash
python3 bonus.py
```

- Prints a precision report (MSE, RMSE, MAE, R²) comparing predictions
  to the actual training data.
- Saves a plot of the data points and the regression line to
  `regression_plot.png`.

The regression line plotted is computed purely from the `theta0`/`theta1`
saved by `train.py`; `matplotlib` is used only to draw it, never to fit
anything.
