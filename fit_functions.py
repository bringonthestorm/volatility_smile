from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
from scipy.optimize import curve_fit
import numpy as np

def square_root_law(x: np.ndarray, a: float, b: float, x0: float):
    """
    Square root law function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a, b, x0 (float): Parameters of the square root law.

    Returns:
    - np.ndarray: Computed values of the square root law.
    """
    return a * np.sqrt(np.maximum(1e-8, x - x0)) + b

def power_law(x: np.ndarray, a: float, b: float, c: float, x0: float):
    """
    Power law function.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a, b, c, x0 (float): Parameters of the power law.

    Returns:
    - np.ndarray: Computed values of the power law.
    """
    return a * (np.maximum(1e-8, x - x0))**c + b

def fit_square_root_law(x_values: np.ndarray, y_values: np.ndarray):
    """
    Fits a square root law model to the given data.

    Parameters:
    - x_values (np.ndarray): Independent variable values.
    - y_values (np.ndarray): Dependent variable values.

    Returns:
    - tuple: Fitted parameters (a, b, x0).
    """
    a = max(y_values) - min(y_values)
    b = min(y_values)
    x0 = min(x_values)
    initial_guess = [a, b, x0]

    return curve_fit(square_root_law, x_values, y_values, p0=initial_guess, maxfev=1000000)[0]

def fit_power_law(x_values: np.ndarray, y_values: np.ndarray):
    """
    Fits a power law model to the given data.

    Parameters:
    - x_values (np.ndarray): Independent variable values.
    - y_values (np.ndarray): Dependent variable values.

    Returns:
    - tuple: Fitted parameters (a, b, c, x0).
    """
    a = max(y_values) - min(y_values)
    b = min(y_values)
    c = 1
    x0 = min(x_values)
    initial_guess = [a, b, c, x0]

    return curve_fit(power_law, x_values, y_values, p0=initial_guess, maxfev=1000000)[0]

def fit_cubic_spline(x_values: np.ndarray, y_values: np.ndarray, s: float = 0.001):
    """
    Fits a cubic spline to the given data.

    Parameters:
    - x_values (np.ndarray): Independent variable values.
    - y_values (np.ndarray): Dependent variable values.
    - s (float): Smoothing factor.

    Returns:
    - UnivariateSpline: Fitted cubic spline model.
    """

    return UnivariateSpline(x=x_values, y=y_values, k=3, s=s, ext=3)

def fit_akima_interpolator(x_values: np.ndarray, y_values: np.ndarray):
    """
    Fits an Akima interpolator to the given data.

    Parameters:
    - x_values (np.ndarray): Independent variable values.
    - y_values (np.ndarray): Dependent variable values.

    Returns:
    - Akima1DInterpolator: Fitted Akima interpolator model.
    """
    return Akima1DInterpolator(x=x_values, y=y_values)

