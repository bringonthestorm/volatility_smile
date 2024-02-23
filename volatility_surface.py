import pandas as pd

from pricing import (
    black_scholes_option_pricing, 
    black_scholes_option_delta
)

from common import (
    filter_dataframe_by_date_token_extrapolation,
    is_date_within_interval,
    calculate_time_to_maturity
)

from maturity_interpolator import MaturityInterpolator
from maturity_extrapolator import MaturityExtrapolator

class VolatilitySurface:
    """
    Class representing a Volatility Surface for an option.
    
    Attributes:
    - filtered_df_nodes (pd.DataFrame): Filtered DataFrame with nodes data.
    - token (str): The token under consideration
    - current_date (str): The current date in 'YYYY-MM-DD HH:MM:SS' format.
    - expiration_date (str): The expiration date of the instrument in 'YYYY-MM-DD HH:MM:SS' format.
    - expiration_dates (np.ndarray): Array of unique expiration dates.
    """

    def __init__(
        self, df_nodes: pd.DataFrame, token: str, current_date: str, interpolation_types: dict, extrapolation_types: dict
    ) -> None:
        """
        Initialize the VolatilitySurface class.

        Parameters:
        - df_nodes (pd.DataFrame): DataFrame containing nodes data.
        - token (str): The token under consideration
        - current_date (str): Current date in 'YYYY-MM-DD HH:MM:SS' format.
        """

        self.token = token
        self.current_date = pd.Timestamp(current_date)
        self.interpolation_types = interpolation_types
        self.extrapolation_types = extrapolation_types
        self.filtered_df_nodes = filter_dataframe_by_date_token_extrapolation(
            df_nodes, token, current_date, extrapolation_types['volatility_curve']
        )
        self.expiration_dates = sorted(self.filtered_df_nodes['expiration_date'].unique().tolist())
        print(f"Expiration Dates: {self.expiration_dates}\n")

        # expiration_date_to_forward_price = df_options.loc[
        #     (df_options['token'] == tokens['target'][0]) & (df_options['current_date'] == current_date), ['expiration_date', 'F']
        # ].drop_duplicates(subset='expiration_date')
        # ir = InterestRate(current_date, expiration_date_to_forward_price)

    def load_expiration_date(self, expiration_date: str) -> None:

        print(type(expiration_date))

        # self.spot_price = spot_price
        self.expiration_date = pd.Timestamp(expiration_date)
        self.time_to_maturity = self._get_time_to_maturity()

        if self._is_expiration_date_present():
            print("-> The expiration date has already been observed. No Interpolation/Extrapolation required.")

            # time to maturity
            # self.time_to_maturity = self.filtered_df_nodes.loc[
            #     self.filtered_df_nodes['expiration_date'] == expiration_date, 'T'
            # ].values[0]
            # forward price
            self.forward_price = self.filtered_df_nodes.loc[
                self.filtered_df_nodes['expiration_date'] == expiration_date, 'F'
            ].values[0]
            # interest rate
            self.interest_rate = self.filtered_df_nodes.loc[
                self.filtered_df_nodes['expiration_date'] == expiration_date, 'r'
            ].values[0]
            # implied volatility fit function
            self.iv_fit_function = self.filtered_df_nodes.loc[
                self.filtered_df_nodes['expiration_date'] == expiration_date, 'func_callable'
            ].values[0]

        else:
            self._prepare_interpolation_or_extrapolation()

        self.spot_price = self.filtered_df_nodes['S'].values[0]
        self.delta_function = self._get_delta_function
        self.call_price_function = self._get_call_price_function

    def _is_expiration_date_present(self) -> bool:
        """
        Check if the expiration date is present in the filtered node data.

        Returns:
        - bool: True if expiration date is present, False otherwise.
        """

        if self.expiration_date in self.expiration_dates:
            return True
        else:
            return False

    def _prepare_interpolation_or_extrapolation(self) -> callable:

        # interpolate
        if is_date_within_interval(self.expiration_dates, self.expiration_date):
            print("-> Interpolation required.")
            interpolator = MaturityInterpolator(self.filtered_df_nodes, self.current_date, self.interpolation_types['volatility_surface'])
            interpolator.load_expiration_date(self.expiration_date)

            self.time_to_maturity = interpolator.time_to_maturity
            self.interest_rate = interpolator.interest_rate
            self.forward_price = interpolator.forward_price
            self.iv_fit_function = interpolator.iv_fit_function

        # extrapolate
        else:
            print("-> Extrapolation required.")
            extrapolator = MaturityExtrapolator(self.filtered_df_nodes, self.current_date, self.extrapolation_types['volatility_surface'])
            extrapolator.load_expiration_date(self.expiration_date)

            self.time_to_maturity = extrapolator.time_to_maturity
            self.interest_rate = extrapolator.interest_rate
            self.forward_price = extrapolator.forward_price
            self.iv_fit_function = extrapolator.iv_fit_function

    
    def _get_time_to_maturity(self) -> float:

        return calculate_time_to_maturity(self.current_date, self.expiration_date)

    # def _get_interest_rate(self) -> callable:

    
    def _get_delta_function(self, strike: float) -> callable:

        iv = self.iv_fit_function(strike)

        return black_scholes_option_delta(
            iv, strike, self.spot_price, self.interest_rate, self.time_to_maturity, 'call'
        )
    
    def _get_call_price_function(self, strike: float) -> callable:

        iv = self.iv_fit_function(strike)

        return black_scholes_option_pricing(
            iv, strike, self.spot_price, self.interest_rate, self.time_to_maturity, 'call'
        )