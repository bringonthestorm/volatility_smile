import pandas as pd
import numpy as np

from pricing import calculate_forward_price
from common import convert_str_to_datetime64, calculate_time_to_maturity


class MaturityExtrapolator:
    """
    This class is used for extrapolating maturity for options based on given data nodes.
    
    Attributes:
        df_nodes (pd.DataFrame): DataFrame containing option data nodes.
        expiration_dates (np.ndarray): Array of unique expiration dates sorted in ascending order.
    """

    def __init__(self, df_nodes: pd.DataFrame, current_date: str, extrapolation_type: str) -> None:
        """
        Initializes the MaturityExtrapolator with data nodes.

        Args:
            df_nodes (pd.DataFrame): DataFrame containing option data nodes.
        """
        self.df_nodes = df_nodes
        self.current_date = current_date
        self.extrapolation_type = extrapolation_type
        self.expiration_dates = sorted(self.df_nodes['expiration_date'].unique().tolist())
        
    def load_expiration_date(self, expiration_date: str) -> None:

        self.expiration_date = expiration_date
        
        # Check if the expiration date is on the left or right side of the surface
        # Then, I fetch either the lower or upper bound of the expiration date
        self._determine_expiration_date_bound()

        self.time_to_maturity = calculate_time_to_maturity(self.current_date, expiration_date)
        self.spot_price = self._get_spot_price()
        self.interest_rate = self._get_interest_rate()
        self.forward_price = calculate_forward_price(self.spot_price, self.interest_rate, self.time_to_maturity)
        self.iv_fit_function = self._get_implied_volatility_fit_function
        
    def _get_spot_price(self) -> float:
        """
        Retrieves the spot price for the determined expiration date bound.
        """
        return self.df_nodes.loc[
            self.df_nodes['expiration_date'] == self.expiration_date_bound, 'S'
        ].values[0]

    def _get_interest_rate(self) -> float:
        """
        Retrieves the interest rate for the determined expiration date bound.
        """
        return self.df_nodes.loc[
            self.df_nodes['expiration_date'] == self.expiration_date_bound, 'r'
        ].values[0]

    def _get_implied_volatility_fit_function(self, strike: float) -> callable:
        """
        Retrieves a callable function that extrapolates the implied volatility for a given expiration date.

        Args:
            expiration_date (str): The expiration date for which the fit function is to be found.
            time_to_maturity (float): Time to maturity for the option.
            extrapolation_type (str): The type of extrapolation method to use.

        Returns:
            callable: A function that computes the extrapolated IV for a given strike price.
        """

        # Retrieves the time to maturity of the closest bound
        self._get_time_to_maturity_bound()

        # Get the IV function for the expiration date on the lower or upper bound
        self._get_implied_volatility_fit_function_bound()

        if self.extrapolation_type == 'flat':
            return self._flat_extrapolation(strike)
        elif self.extrapolation_type == 'square_root':
            return self._square_root_of_time_extrapolation(strike)

    def _determine_expiration_date_bound(self) -> None:
        """
        Determines whether the expiration date is beyond the available data range and sets the closest bound.
        """
        lower_bound = self.expiration_dates[0]
        upper_bound = self.expiration_dates[-1]

        expiration_date_dt64 = convert_str_to_datetime64(self.expiration_date)

        if expiration_date_dt64 < lower_bound:
            self.expiration_date_bound = lower_bound
        elif expiration_date_dt64 > upper_bound:
            self.expiration_date_bound = upper_bound
        else:
            raise ValueError("Expiration date is within bounds; consider interpolation instead.")
    
    def _get_time_to_maturity_bound(self) -> None:
        """
        Retrieves the time to maturity for the determined expiration date bound.
        """
        self.time_to_maturity_bound = self.df_nodes.loc[
            self.df_nodes['expiration_date'] == self.expiration_date_bound, 'T'
        ].values[0]

    def _get_implied_volatility_fit_function_bound(self) -> None:
        """
        Retrieves the implied volatility fit function for the determined expiration date bound.
        """
        self.iv_fit_function_bound = self.df_nodes.loc[
            self.df_nodes['expiration_date'] == self.expiration_date_bound, 'func_callable'
        ].values[0]

    # Square Root of Time - Model
    # ---------------------------
    def _square_root_of_time_extrapolation(self, strike: float) -> float:
        """
        Implements the square root of time model to extrapolate implied volatility.

        Args:
            strike (float): The strike price for which to calculate implied volatility.

        Returns:
            float: The extrapolated implied volatility for the given strike price.
        """
        forward_price_bound = calculate_forward_price(self.spot_price, self.interest_rate, self.time_to_maturity_bound)

        strike_bound = forward_price_bound * (strike / self.forward_price)**np.sqrt(self.time_to_maturity_bound / self.time_to_maturity)

        return self.iv_fit_function_bound(strike_bound)

    def _flat_extrapolation(self, strike: float) -> float:

        return self.iv_fit_function_bound(strike)
