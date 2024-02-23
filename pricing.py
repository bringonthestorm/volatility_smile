from scipy.stats import norm
from typing import Union
import numpy as np

def black_scholes_option_pricing(vol: float, K: float, S: float, r: float, T: float, option_type: str) -> float:
    """
    Calculates the Black and Scholes option pricing for European options.

    Parameters:
    - vol (float): Volatility of the underlying asset.
    - K (float): Strike price of the option.
    - S (float): Spot price of the underlying asset.
    - r (float): Risk-free interest rate.
    - T (float): Time to maturity of the option.
    - option_type (str): Type of option, either 'call' or 'put'.

    Returns:
    - float: Option price.
    """

    def d1():
        """
        Calculates the d1 parameter used in the Black and Scholes formula.
        """
        return (np.log(S/K) + (r+0.5*(vol**2))*T) / (vol * np.sqrt(T))

    def d2():
        """
        Calculates the d2 parameter used in the Black and Scholes formula.
        """
        return d1() - (vol * np.sqrt(T))

    def option_price():
        """
        Calculates the Black and Scholes option price based on the option type.
        """
        return (
            option_sign * (
                S * norm.cdf(option_sign * d1()) - K * np.exp(-r * T) * norm.cdf(option_sign * d2())
            )
        )

    if option_type.lower() == 'call':
        option_sign = 1
    elif option_type.lower() == 'put':
        option_sign = -1
    else:
        raise ValueError("Option type can only be 'call' or 'put'.")

    return option_price()


def black_scholes_option_delta(vol: float, K: float, S: float, r: float, T: float, option_type: str) -> float:
    """
    Calculates the Black and Scholes delta for European options.

    Parameters:
    - vol (float): Volatility of the underlying asset.
    - K (float): Strike price of the option.
    - S (float): Forward price of the underlying asset.
    - r (float): Risk-free interest rate.
    - T (float): Time to maturity of the option.
    - option_type (str): Type of option, either 'call' or 'put'.

    Returns:
    - float: Delta of the option.
    """

    def d1():
        """
        Calculates the d1 parameter used in the Black and Scholes formula.
        """
        return (np.log(S/K) + (r+0.5*(vol**2))*T) / (vol * np.sqrt(T))

    def option_delta():
        """
        Calculates the Black and Scholes delta based on the option type.
        """
        return (
            option_sign * norm.cdf(option_sign * d1())
        )

    if option_type.lower() == 'call':
        option_sign = 1
    elif option_type.lower() == 'put':
        option_sign = -1
    else:
        raise ValueError("Option type can only be 'call' or 'put'.")

    return option_delta()


def calculate_forward_price(
    S: Union[float, np.ndarray], r: Union[float, np.ndarray], T: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculates the forward price of an asset.

    Parameters:
    - S: Current spot price of the asset. Can be a float or a numpy array.
    - r: Risk-free interest rate. Can be a float or a numpy array.
    - T: Time to maturity for the forward contract. Can be a float or a numpy array.

    Returns:
    - Union[float, np.ndarray]: Forward price of the asset.
    """
    
    # return the Forward Price
    return S * np.exp(r * T)


def calculate_strike_with_black_scholes_delta(
    vol: Union[float, np.ndarray], S: Union[float, np.ndarray], r: Union[float, np.ndarray],
    T: Union[float, np.ndarray], delta: Union[float, np.ndarray], option_type = str
) -> Union[float, np.ndarray]:

    """
    Calculate the strike price using the Black-Scholes delta expression.
    
    Parameters:
    - vol (float or np.ndarray): Volatility of the underlying asset.
    - S (float or np.ndarray): Current spot price of the underlying asset.
    - r (float or np.ndarray): Risk-free interest rate.
    - T (float or np.ndarray): Time to maturity of the option.
    - delta (float or np.ndarray): Delta value representing the option sensitivity to changes in spot price.
    - option_type (str): Type of the option (e.g., 'call' or 'put').

    Returns:
    - (float or np.ndarray): Strike price based on the given parameters.
    """
    
    # Calculate the inverse of the cumulative distribution function (CDF) at the specified delta
    inverse_norm = norm.ppf(delta)

    # Calculate the exponent term in the Black-Scholes formula
    exponent = vol * np.sqrt(T) * inverse_norm - (r + 0.5 * (vol ** 2)) * T

    # Return the strike price
    return S * np.exp(-exponent)


