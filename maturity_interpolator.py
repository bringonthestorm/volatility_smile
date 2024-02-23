import pandas as pd
import numpy as np

from common import (
    calculate_time_to_maturity,
    find_strict_nearest_date_bounds
)

class MaturityInterpolator:
    """
    This class is used for interpolating maturity for options based on given data nodes.

    Attributes:
        df_nodes (pd.DataFrame): DataFrame containing option data nodes.
        expiration_dates (np.ndarray): Array of unique expiration dates sorted in ascending order.
    """

    def __init__(self, df_nodes: pd.DataFrame, current_date: str, interpolation_type: str) -> None:

        """
        Initializes the MaturityInterpolator with data nodes.

        Args:
            df_nodes (pd.DataFrame): DataFrame containing option data nodes.
        """
        self.df_nodes = df_nodes
        self.current_date = current_date
        self.interpolation_type = interpolation_type
        self.expiration_dates = sorted(df_nodes['expiration_date'].unique().tolist())

    def load_expiration_date(self, expiration_date: str) -> None:

        self.expiration_date = pd.Timestamp(expiration_date)

        self.time_to_maturity = calculate_time_to_maturity(self.current_date, expiration_date)
        self.spot_price = self._get_spot_price()
        self.forward_price = self._get_forward_price()
        self.interest_rate = self._get_interest_rate()
        self.iv_fit_function = self._get_implied_volatility_fit_function

    def _get_spot_price(self) -> float:
        return self.df_nodes['S'].values[0]

    def _get_forward_price(self) -> float:

        # forward price (linear interpolation)
        time_to_maturity_values = self.df_nodes['T'].values
        forward_price_values = self.df_nodes['F'].values

        return np.interp(self.time_to_maturity, time_to_maturity_values, forward_price_values)

    def _get_interest_rate(self) -> float:

        return np.log(self.forward_price/self.spot_price) / self.time_to_maturity

    def _get_implied_volatility_fit_function(self, strike: float) -> float:
        """
        Retrieves a callable function that interpolates the implied volatility for a given expiration date.

        Returns:
            float: Immplied volatility for a given strike price.
        """

        # Determines the lower and upper bounds of expiration dates closest to the expiration date.
        self.expiration_date_bounds = find_strict_nearest_date_bounds(
            self.expiration_dates, self.expiration_date
        )

        # Get the time to maturity for the lower and upper bound expiration dates
        self._get_time_to_maturity_bounds()

        # 3rd - Get the IV functions for the expiration dates on the lower and upper bounds
        self._get_implied_volatility_fit_function_bounds()

        # 4th - Return callable (with interpolation)
        if None not in self.expiration_date_bounds:
            if self.interpolation_type == 'linear':
                return self.linear_interpolation(strike)
            elif self.interpolation_type == 'flat_forward':
                return self.flat_forward_interpolation(strike)
        else:
            raise ValueError("Invalid expiration date bounds. Data may be missing for interpolation.")

    def _get_time_to_maturity_bounds(self) -> None:
        """ Retrieves the time to maturity for the lower and upper bound expiration dates. """
        self.time_to_maturity_bounds = [
            self.df_nodes.loc[self.df_nodes['expiration_date'] == d, 'T'].values[0] if d is not None else None for d in self.expiration_date_bounds
        ]

    def _get_implied_volatility_fit_function_bounds(self) -> callable:
        """ Retrieves the IV functions for the lower and upper bound expiration dates. """
        self.iv_fit_function_bounds = self.df_nodes.loc[
            self.df_nodes['expiration_date'].isin(self.expiration_date_bounds), 'func_callable'
        ].tolist()

    # Linear Volatility Interpolation - Model
    # ---------------------------------------
    def linear_interpolation(self, strike: float) -> float:
        """
        Performs linear interpolation between the IV functions at the bounds.

        Args:
            strike (float): The strike price for which to calculate implied volatility.

        Returns:
            float: The interpolated value of the implied volatility.
        """

        weight = calculate_time_weight(
            self.expiration_date_bounds[0], self.expiration_date_bounds[1], self.expiration_date
        )

        return (
            weight * self.iv_fit_function_bounds[0](strike) 
            + 
            (1-weight) * self.iv_fit_function_bounds[1](strike)
        )

    # Flat Forward Volatility Interpolation - Model
    # ---------------------------------------
    def flat_forward_interpolation(self, strike: float) -> float:
        """
        Performs flat forward interpolation between the IV functions at the bounds.

        Args:
            strike (float): The strike price for which to calculate implied volatility.

        Returns:
            float: The interpolated value of the implied volatility.
        """

        # compute forward IV
        forward_iv = np.sqrt(
            (self.iv_fit_function_bounds[1](strike)**2 * self.time_to_maturity_bounds[1] - self.iv_fit_function_bounds[0](strike)**2 * self.time_to_maturity_bounds[0]) 
            / 
            (self.time_to_maturity_bounds[1] - self.time_to_maturity_bounds[0])
        )

        return np.sqrt(
            (self.iv_fit_function_bounds[0](strike)**2 * self.time_to_maturity_bounds[0] + forward_iv**2 * (self.time_to_maturity - self.time_to_maturity_bounds[0])) 
            / 
            self.time_to_maturity
        )




def calculate_time_weight(start_date: str, end_date: str, target_date: str) -> float:
    """
    Calculates the weight of a target date within a given time interval.

    The weight represents the proportion of the time interval from start_date to end_date
    that has elapsed by the target_date. A weight of 0 means the target_date is at start_date,
    while a weight of 1 means it is at end_date.

    Args:
        start_date (str): The start date of the interval in 'YYYY-MM-DD HH:MM:SS' format.
        end_date (str): The end date of the interval in 'YYYY-MM-DD HH:MM:SS' format.
        target_date (str): The target date to be weighted, in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        float: The weight of the target_date within the specified time interval.

    Raises:
        ValueError: If end_date is earlier than start_date or if target_date is outside the interval.

    # Example Usage
    # --------------------------------------------------------------------------#

    start_date = '2023-12-01 08:00:00'
    end_date = '2023-12-05 08:00:00'
    target_date = '2023-12-03 15:00:00'

    time_weight = calculate_time_weight(start_date, end_date, target_date)
    print(f"Time Weight: {time_weight}")
    Time Weight: 0.5729166666666666
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    target = pd.Timestamp(target_date)

    if end < start:
        raise ValueError("End date must be after start date.")
    if target < start or target > end:
        raise ValueError("Target date must be within the start and end dates.")

    return (end - target) / (end - start)