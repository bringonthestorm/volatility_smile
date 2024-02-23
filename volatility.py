import numpy as np
from scipy.optimize import bisect

from pricing import black_scholes_option_pricing
from fit_functions import (
    square_root_law, 
    power_law, 
    fit_square_root_law, 
    fit_power_law, 
    fit_cubic_spline,
    fit_akima_interpolator
)

def implied_volatility(K: float, S: float, r: float, T: float, option_price: float, option_type: str) -> float:
    """
    Calculate the implied volatility for an option using the Black-Scholes model.

    Parameters:
    - K (float): Strike price.
    - S (float): Spot price of the underlying asset.
    - r (float): Risk-free interest rate.
    - T (float): Time to expiration in years.
    - option_price (float): Current market price of the option.
    - option_type (str): Type of the option ('call' or 'put').

    Returns:
    - float: Implied volatility.
    """
    def black_scholes_difference(vol):
        return black_scholes_option_pricing(vol, K, S, r, T, option_type) - option_price

    return bisect(black_scholes_difference, -0.01, 5, xtol=1e-8)


class StrikeVolatilityModelling:
    """
    A class to model the relationship between strike prices and implied volatilities 
    using cubic or akima spline interpolation and optional extrapolation methods for tail behavior.
    
    Attributes:
        strike_values (np.ndarray): Array of strike prices.
        volatility_values (np.ndarray): Array of corresponding implied volatilities.
        spot_price (float): Current price of the underlying asset.
        extrapolation_type (int): Indicator of the method used for tails extrapolation.
    """

    def __init__(
        self, strike_values: np.ndarray, volatility_values: np.ndarray, spot_price: float, extrapolation_type: int='flat'
    ) -> None:
        self.strike_values = strike_values
        self.volatility_values = volatility_values
        self.spot_price = spot_price
        self.extrapolation_type = extrapolation_type

        self.max_strike = np.max(strike_values)
        self.min_strike = np.min(strike_values)
        self.strike_at_min_volatility = strike_values[np.argmin(volatility_values)]

        self.first_volatility = volatility_values[0]
        self.last_volatility = volatility_values[-1]

        # Fit spline to strike/implied volatility (IV) values
        self.spline = self._fit_spline_model(model='cubic')

        if extrapolation_type in ['square_root_law', 'power_law']:
            self._extrapolate_tails()

    def _fit_spline_model(self, model: str, s: float=0) -> callable:
        """
        Fits a cubic or akima spline to the strike and volatility values.
        
        Returns:
            Spline: A cubic or Akima spline fitted to strike and volatility values.
        """
        if model == 'cubic':
            return fit_cubic_spline(self.strike_values, self.volatility_values, s=s)

        elif model == 'akima':
            return fit_akima_interpolator(self.strike_values, self.volatility_values)


    def _extrapolate_tails(self) -> None:
        """
        Performs extrapolation for the tails using either square root or power law methods.
        """
        sample_strikes, sample_vols = self._sample_strike_volatility_values()
        sample_strikes_rt, sample_vols_rt = self._get_right_tail_values(sample_strikes, sample_vols)

        if self.extrapolation_type == 'square_root_law':
            self.square_root_law_params = fit_square_root_law(sample_strikes_rt, sample_vols_rt)
        elif self.extrapolation_type == 'power_law':
            self.power_law_params = fit_power_law(sample_strikes_rt, sample_vols_rt)

        self._update_extrapolated_values()

    def _sample_strike_volatility_values(self) -> (np.ndarray, np.ndarray):
        """
        Generates a sample of strike and volatility values from the spline.

        Returns:
            tuple: A tuple containing arrays of sample strike values and corresponding volatilities.
        """
        sample_strikes = np.linspace(self.min_strike, self.max_strike, 20)
        sample_vols = np.array([self.spline(strike) for strike in sample_strikes])
        return sample_strikes, sample_vols

    def _get_right_tail_values(self, strikes: np.ndarray, volatilities: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Extracts the right tail values from the given strike and volatility arrays.

        Args:
            strikes (np.ndarray): Array of strike values.
            volatilities (np.ndarray): Array of volatility values.

        Returns:
            tuple: Right tail strike and volatility values.
        """
        min_volatility_index = np.argmin(volatilities)
        return strikes[min_volatility_index:], volatilities[min_volatility_index:]

    def _update_extrapolated_values(self) -> None:
        """
        Updates the class with the extrapolated strike and volatility values.
        """
        extrapolated_strikes_rt = self._get_extrapolation_strikes()
        extrapolated_vols_rt = np.array([self.get_iv_for_strike(strike) for strike in extrapolated_strikes_rt])

        valid_indices = extrapolated_vols_rt >= self.last_volatility
        self.extrapolated_strikes_rt = extrapolated_strikes_rt[valid_indices]
        self.extrapolated_vols_rt = extrapolated_vols_rt[valid_indices]

        self.strike_values = np.concatenate((self.strike_values, self.extrapolated_strikes_rt))
        self.volatility_values = np.concatenate((self.volatility_values, self.extrapolated_vols_rt))

        if self.extrapolation_type in ['square_root_law', 'power_law']:
            # update min/max strike values
            self.max_strike = np.max(self.strike_values)
            self.min_strike = np.min(self.strike_values)
            # Re-fit the spline with updated values
            self.spline = self._fit_spline_model(model='cubic', s = 0.01)

    def _get_extrapolation_strikes(self) -> np.ndarray:
        """
        Generates strike values for right tail extrapolation.

        Returns:
            np.ndarray: Array of strike values for extrapolation.
        """
        max_multiplier = 5
        extra_strikes = np.linspace(self.spot_price, max_multiplier * self.spot_price, 20)
        return extra_strikes[extra_strikes > self.max_strike]

    def get_iv_for_strike(self, strike: float) -> float:
        """
        Returns the implied volatility for a given strike price.

        Args:
            strike (float): The strike price.

        Returns:
            float: The implied volatility for the given strike price.
        """
        if strike <= self.max_strike and strike >= self.min_strike:

            volatility = self.spline(strike)

            if self.extrapolation_type == 'flat':
                if volatility > self.last_volatility and strike > self.strike_at_min_volatility:
                    volatility = self.last_volatility
                elif volatility > self.first_volatility and strike < self.strike_at_min_volatility:
                    volatility = self.first_volatility

            return volatility.item()

        if self.extrapolation_type == 'square_root_law' and strike > self.max_strike:
            return square_root_law(strike, *self.square_root_law_params)
        elif self.extrapolation_type == 'power_law' and strike > self.max_strike:
            return power_law(strike, *self.power_law_params)

        return self.first_volatility if strike < self.min_strike else self.last_volatility