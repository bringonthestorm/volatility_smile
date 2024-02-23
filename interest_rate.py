import pandas as pd
import numpy as np

from common import (
    calculate_time_to_maturity,
    is_date_within_interval
)

def implied_interest_rate(spot_price: float, forward_price: float, time_to_maturity: float) -> float:
    """
    Calculate the implied interest rate based on the spot price, forward price, and time to maturity.

    'desire for leverage'

    Parameters:
    - spot_price (float): The current spot price of the asset.
    - forward_price (float): The forward price of the asset for a specific expiration date.
    - time_to_maturity (float): The time to maturity in years.

    Returns:
    - float: The calculated implied interest rate.
    """
    return np.log(forward_price / spot_price) / time_to_maturity

class InterestRate:
    def __init__(self, current_date: str, expiration_date_to_forward_price: pd.DataFrame) -> None:
        """
        Initialize an InterestRate object to calculate interest rates for different expiration dates.

        Parameters:
        - current_date (str): The current date in 'YYYY-MM-DD HH:MM:SS' format.
        - expiration_date_to_forward_price (pd.DataFrame): 
            - A DataFrame mapping expiration dates ('YYYY-MM-DD HH:MM:SS') to their respective forward prices.
            - The DataFrame should have columns ['expiration_date', 'F'].

        if tte = TTE_observed: 
            -> compute interest rate with the forward formula using the observed values.
        if TTE_min < tte < TTE_max and tte != TTE_observed: 
            -> interpolate forward price, and then compute the interest rate with the forward formula.
        if tte > TTE_max: 
            -> interest_rate = interest_rate(TTE_max).
        if tte < TTE_min: 
            -> interest_rate = interest_rate(TTE_min).

        """
        self.current_date = pd.Timestamp(current_date)
        self.expiration_date_to_forward_price = expiration_date_to_forward_price.sort_values(by='expiration_date')
        self.expiration_dates = self.expiration_date_to_forward_price['expiration_date'].tolist()
        self.forward_prices = self.expiration_date_to_forward_price['F'].tolist()
        self.interest_rates = self.expiration_date_to_forward_price['r'].tolist()
        self.time_to_maturities = [
            calculate_time_to_maturity(self.current_date, expiration_date) for expiration_date in self.expiration_dates
        ]

    def calculate_interest_rate(self, spot_price: float, expiration_date: str) -> float:
        """
        Calculate the interest rate based on the spot price and a given expiration date.

        Parameters:
        - spot_price (float): The current spot price of the asset.
        - expiration_date (str): The expiration date for which to calculate the interest rate ('YYYY-MM-DD').

        Returns:
        - float: The calculated interest rate.
        """
        expiration_date = pd.Timestamp(expiration_date)
        time_to_maturity = calculate_time_to_maturity(self.current_date, expiration_date)

        try:
            forward_price = self._get_forward_price(expiration_date)
            return implied_interest_rate(spot_price, forward_price, time_to_maturity)
        except ValueError:
            if is_date_within_interval(self.expiration_dates, expiration_date):
                # forward_price = self._interpolate_forward_price(time_to_maturity)
                # return implied_interest_rate(spot_price, forward_price, time_to_maturity)

                return np.interp(time_to_maturity, self.time_to_maturities, self.interest_rates)

            elif time_to_maturity < self.time_to_maturities[0]:
                return implied_interest_rate(spot_price, self.forward_prices[0], self.time_to_maturities[0])
            else:
                return implied_interest_rate(spot_price, self.forward_prices[-1], self.time_to_maturities[-1])

    def _get_forward_price(self, expiration_date: pd.Timestamp) -> float:
        """
        Get the forward price for a given expiration date.

        Parameters:
        - expiration_date (pd.Timestamp): The expiration date.

        Returns:
        - float: The forward price for the given expiration date.

        Raises:
        - ValueError: If no forward price is found for the given expiration date.
        """
        price = self.expiration_date_to_forward_price.loc[
            self.expiration_date_to_forward_price['expiration_date'] == expiration_date, 'F'
        ].values
        if len(price) == 0:
            raise ValueError(f"No forward price found for expiration date: {expiration_date}")
        return price[0]

    def _interpolate_forward_price(self, time_to_maturity: float) -> float:
        """
        Interpolate the forward price based on the time to maturity.

        Parameters:
        - time_to_maturity (float): The time to maturity in years.

        Returns:
        - float: The interpolated forward price.
        """
        return np.interp(time_to_maturity, self.time_to_maturities, self.forward_prices)
