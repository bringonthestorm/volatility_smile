from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from loaders.ch_options_chain import (
    get_tardis_options_chain_1m_unique_timestamps,
    get_tardis_options_chain_1m_from_timestamps,
    get_tardis_options_chain_1m
)
from loaders.ch_volatility import get_realized_volatility_1m
from loaders.ch_mdc_public_trades import get_ag_mdc_public_trades_ohlc_1m

from volatility import implied_volatility
from pricing import black_scholes_option_delta
from common import find_start_times, calculate_time_to_maturity
from interest_rate import implied_interest_rate

class SourceOptionsDataProcessor:
    """
    Processes options data for a given time frame and set of instruments.

    Attributes:
        start_date (str): Start date for data fetching in 'YYYY-MM-DD HH:MM:SS' format.
        end_date (str): End date for data fetching in 'YYYY-MM-DD HH:MM:SS' format.
        tokens (list[str]): List of tokens for aquiring option chain data.
        symbols (list[str]): List of symbols for aquiring the spot price.
        instrument_ids (list[str]): List of instrument IDs for aquiring the realized volatility.
        open_prices_only (bool): If True, filters data to include only opening prices.
        df (pd.DataFrame): DataFrame containing processed source options data.

    Methods:
        process_data: Main method to initiate the processing of options data.
    """

    def __init__(
        self, start_date: str, end_date: str,
        tokens: list[str], symbols: list[str], instrument_ids: list[str],
        period: str='day'
    ) -> None:

        self.start_date = start_date
        self.end_date = end_date
        self.tokens = tokens
        self.symbols = symbols
        self.instrument_ids = instrument_ids
        self.period = period

        self.rolling_window = '6_month' # realized volatility

        if period in ['day', 'hour', 'minute']:
            df_options_timestamps = get_tardis_options_chain_1m_unique_timestamps(
                start_date, end_date, tokens
            )
            options_timestamps = list(df_options_timestamps['ts'].values)
            options_timestamps = find_start_times(options_timestamps, self.period)

            self.df = get_tardis_options_chain_1m_from_timestamps(tokens, options_timestamps)
        else:
            self.df = get_tardis_options_chain_1m(start_date, end_date, tokens)
        
        self.process_data()

    def process_data(self) -> None:

        self._rename_columns()
        self._add_spot_price()
        self._add_time_to_maturity()
        self._add_implied_interest_rate()
        self._filter_out_noisy_observations()
        self._convert_option_price_to_usd()
        self._add_realized_volatility()
        self._add_implied_volatility_and_delta()
        self._drop_unnecessary_columns()
        
        # self._ensure_volatility_smile_convexity()

    def _rename_columns(self) -> None:
        """
        Renames columns in the DataFrame for better readability
        """
        rename_dict = {
            'ts': 'current_date',
            'underlying_asset': 'token',
            'expiration': 'expiration_date',
            'symbol': 'option_contract',
            'type': 'option_type',
            'strike_price': 'K',
            'underlying_index': 'futures_contract',
            'forward_price': 'F',
            'mark_price': 'option_price',
            'spot_price': 'S'
        }

        self.df.rename(columns=rename_dict, inplace=True)

    def _add_spot_price(self) -> None:
        """
        Adds a new column to the dataframe with the spot prices for a given "current_date" value.
        """

        df_spot_price = get_ag_mdc_public_trades_ohlc_1m(self.start_date, self.end_date, self.symbols)

        df_spot_price.rename(
            columns={'ts': 'current_date', 'underlying_asset': 'token', 'spot_price': 'S'}, 
            inplace=True
        )

        # For a given 'current_date' and 'token', the respective spot price is added.
        self.df = pd.merge(
            self.df, df_spot_price[['current_date', 'token', 'S']], 
            on = ['current_date', 'token'], how = 'left'
        )

    def _add_time_to_maturity(self) -> None:
        """
        Adds a new column to the dataframe with the Time to Maturity (in years).
        """

        self.df['T'] = [calculate_time_to_maturity(
            row['current_date'], row['expiration_date']
        ) for i, row in self.df.iterrows()]

    def _add_implied_interest_rate(self) -> None:
        """
        Adds a new column to the dataframe with the Implied Interest Rate.

        The interest rate between the forward and the spot price.
        """

        self.df['r'] = [implied_interest_rate(
            row['S'], row['F'], row['T']
        ) for i, row in self.df.iterrows()]
        

    def _filter_out_noisy_observations(self) -> None:
        """
        Removes rows from the dataframe according to a given criteria.

        1. Remove Out of the Money Options.
        2. Remove options for which the forward price was computed synthetically.
        3. Remove options with a option price lower or equal to 2 bps.
        4. Remove options with a Time to Maturity less or equal to 2 days.
        """

        # remove In the Money options
        self.df = remove_imo_options(self.df)

        # remove observations where the foward price was computed synthetically with options data
        self.df = self.df[~self.df['futures_contract'].str.startswith('SYN.')] 

        # only keep observations where the option_price is higher than 2bps
        self.df = self.df[self.df['option_price'] > 0.002] 

        # only keep observations with Time to Expiration higher than 7 days
        self.df = self.df[self.df['T'] > 7/365] 

        # only keep observations where the open_interest is greater than zero.
        self.df = self.df[self.df['open_interest'] > 0] 

        # only keep observations where the bid_price is higher than 5bps
        self.df = self.df[self.df['bid_price'] > 0.001] 

        # only keep observations where the ask_price is higher than 5bps
        self.df = self.df[self.df['ask_price'] > 0.001] 

    def _convert_option_price_to_usd(self) -> None:

        self.df['option_price'] = self.df['option_price'] * self.df['F']

    def _add_realized_volatility(self) -> None:

        """
        Adds a new column to the dataframe with the Annualized Realized Volatility.
        """
        # Load Realized Volatility values for Source Instruments
        start_date_str = (datetime.strptime(self.start_date, '%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d')
        end_date_str = (datetime.strptime(self.end_date, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)).strftime('%Y-%m-%d')

        df_realized_volatility = get_realized_volatility_1m(
            start_date_str, end_date_str, self.instrument_ids, self.rolling_window, self.period
        )

        df_realized_volatility.rename(columns={
            'ts': 'current_date_no_time',
            'base_currency': 'token',
            'annualized_realized_volatility': 'rv'
        }, inplace=True)

        # Create a new column with the current_date without the time
        self.df['current_date_no_time'] = pd.to_datetime(self.df['current_date'].dt.strftime('%Y-%m-%d'))

        # For a given 'current_date' and 'token', the respective annualized realized volatility is added.
        self.df = pd.merge(
            self.df, 
            df_realized_volatility[['current_date_no_time', 'token', 'rv']], 
            on=['current_date_no_time', 'token'], how='left'
        )

        self.df.drop('current_date_no_time', axis=1, inplace=True)

    def _add_implied_volatility_and_delta(self) -> None:

        iv_values = []
        delta_values = []

        for _, row in self.df.iterrows():

            iv = implied_volatility(
                row['K'], row['S'], row['r'], row['T'], row['option_price'], row['option_type']
            )
            delta = black_scholes_option_delta(
                iv, row['K'], row['S'], row['r'], row['T'], 'call'
            )
            iv_values.append(iv)
            delta_values.append(delta)

        self.df['iv'] = iv_values
        self.df['delta'] = delta_values


    def _drop_unnecessary_columns(self):
        columns_to_drop = [
            'option_contract', 'option_type', 
            'futures_contract', 'open_interest', 
            'bid_price', 'ask_price'
        ]

        self.df.drop(columns=columns_to_drop, axis=1, inplace=True)


    def _ensure_volatility_smile_convexity(self) -> None:
        """
        Ensure the convexity of the volatility smile across different options.

        This function iterates through unique combinations of current dates, tokens, 
        and expiration dates in the provided dataframe. For each combination, it 
        extracts implied volatility (IV) values and adjusts them to maintain convexity 
        of the volatility smile curve.
        """

        if not set(['current_date', 'token', 'expiration_date', 'iv']).issubset(self.df.columns):
            raise ValueError("DataFrame must contain 'current_date', 'token', 'expiration_date', and 'iv' columns.")

        unique_combinations = self.df[['current_date', 'token', 'expiration_date']].drop_duplicates()
        updated_dataframes = []

        for _, combination in unique_combinations.iterrows():
            current_date, token, expiration_date = combination

            filtered_df = self.df.loc[
                (self.df['current_date'] == current_date) &
                (self.df['token'] == token) &
                (self.df['expiration_date'] == expiration_date)
            ]

            iv_values = filtered_df['iv'].values

            convex_values, indices = make_convex_with_min(iv_values)
            updated_dataframes.append(filtered_df.iloc[indices])

        self.df = pd.concat(updated_dataframes)


# Auxiliary Functions
# -------------------
def remove_imo_options(df: pd.DataFrame):
    """
    Remove in the money options (according to their strike and type).

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'K', 'F', and 'option_type' columns.

    Returns:
    - pd.DataFrame: Filtered DataFrame with relevant options.
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['K', 'F', 'option_type']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("DataFrame is missing required columns.")

    # Handle edge cases where 'K' or 'F' values might be missing or NaN
    df = df.dropna(subset=['K', 'F'])

    # Filter options based on the relationship between 'K' and 'F'
    put_filter = (df['K'] < df['F']) & (df['option_type'] == 'put')
    call_filter = (df['K'] >= df['F']) & (df['option_type'] == 'call')

    return df.loc[put_filter | call_filter]

def make_convex_with_min(array: np.array) -> (np.array, np.array):
    """
    Transforms the given array into a convex shape with a single global minimum.

    The function identifies the global minimum and then ensures that the array is
    non-decreasing to the left of the minimum and non-increasing to the right. 
    The result is a convex array centered around the global minimum.
    
    Parameters:
        array (np.array): The input array to be transformed.

    Returns:
        convex_array (np.array): The transformed convex array.
        kept_indices (np.array): The indices of the original array that were kept.
    """
    # Find the index of the global minimum
    min_index = np.argmin(array)

    # Initialize the new array with the first element and its index
    convex_array = [array[0]]
    kept_indices = [0]

    # Iterate over the array to maintain convexity
    for i in range(1, len(array)):
        curr_val = convex_array[-1]

        # Check conditions for maintaining convexity
        if ((i < min_index) and (array[i] < curr_val)) or \
           ((i > min_index) and (array[i] > curr_val)) or \
           (i == min_index):
            convex_array.append(array[i])
            kept_indices.append(i)

    return np.array(convex_array), np.array(kept_indices)