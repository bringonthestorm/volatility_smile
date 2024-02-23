from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from loaders.ch_mdc_public_trades import get_ag_mdc_public_trades_ohlc_1m
from loaders.ch_volatility import get_realized_volatility_1m
from pricing import (
    calculate_forward_price,
    calculate_strike_with_black_scholes_delta,
    black_scholes_option_pricing
)

class TargetOptionsDataframe:

    def __init__(
        self, start_date: str, end_date: str, source_token: str, 
        target_tokens: list[str], target_symbols: list[str], target_instrument_ids: list[str], 
        df_source_options: pd.DataFrame, period: str='day'
    ) -> None:

        self.start_date = start_date
        self.end_date = end_date
        self.source_token = source_token
        self.target_tokens = target_tokens
        self.target_symbols = target_symbols
        self.target_instrument_ids = target_instrument_ids

        self.rolling_window = '6_month'
        self.period = period

        self.df_source_options = df_source_options.loc[
            df_source_options['token'] == source_token
        ]

        self.df = self._compute_initial_dataframe()

        self._add_spot_price()
        self._add_forward_price()
        self._add_realized_volatility()
        self._scale_implied_volatility()
        self._add_strike_price()
        self._add_option_price()

        self.df.drop('rv_source', axis=1, inplace=True)

        return None

    def _compute_initial_dataframe(self) -> None:

        dataframes_list = []

        for target_token in self.target_tokens:
            
            df_tmp = self.df_source_options.copy(deep=True)

            df_tmp.drop('S', axis=1, inplace=True)
            df_tmp.drop('F', axis=1, inplace=True)
            df_tmp.drop('K', axis=1, inplace=True)

            df_tmp['token'] = target_token

            df_tmp['option_price'] = np.nan
        
            df_tmp.rename(columns={'rv': 'rv_source'}, inplace=True)

            dataframes_list.append(df_tmp)

        return pd.concat(dataframes_list).reset_index(drop=True)

    
    def _add_spot_price(self) -> None:
        """
        Adds a new column to the dataframe with the "spot price".
        """

        df_spot_price = get_ag_mdc_public_trades_ohlc_1m(self.start_date, self.end_date, self.target_symbols)

        df_spot_price.rename(columns={
            'ts': 'current_date',
            'underlying_asset': 'token',
            'spot_price': 'S'
        }, inplace=True)

        # For a given 'current_date' and 'token', the respective spot price is added.
        self.df = pd.merge(
            self.df, 
            df_spot_price[['current_date', 'token', 'S']], 
            on=['current_date', 'token'], how='left'
        )

        return None
    
    def _add_forward_price(self) -> None:
        """
        Adds a new column to the dataframe with the "forward price".
        """

        self.df['F'] = calculate_forward_price(
            S = self.df['S'].values,
            r = self.df['r'].values,
            T = self.df['T'].values
        )

        return None

    def _add_realized_volatility(self) -> None:

        """
        Adds a new column to the dataframe with the Annualized Realized Volatility.
        """
        # Load Realized Volatility values for Source Instruments
        start_date_str = (datetime.strptime(self.start_date, '%Y-%m-%d %H:%M:%S')).strftime('%Y-%m-%d')
        end_date_str = (datetime.strptime(self.end_date, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)).strftime('%Y-%m-%d')

        df_realized_volatility = get_realized_volatility_1m(
            start_date_str, end_date_str, self.target_instrument_ids, self.rolling_window, self.period
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

        return None

    def _scale_implied_volatility(self) -> None:

        self.df['iv'] = self.df['iv'].values * self.df['rv'].values / self.df['rv_source'].values

        return None

    def _add_strike_price(self) -> None:
        """
        Adds a new column to the dataframe with the "strike price".
        """

        self.df['K'] = calculate_strike_with_black_scholes_delta(
            vol = self.df['iv'].values,
            S = self.df['S'].values,
            r = self.df['r'].values,
            T = self.df['T'].values,
            delta = self.df['delta'].values,
            option_type = 'call'
        )

        return None

    def _add_option_price(self) -> None:
        """
        Adds a new column to the dataframe with the "option price".
        """

        self.df['option_price'] = black_scholes_option_pricing(
            vol = self.df['iv'].values,
            K = self.df['K'].values,
            S = self.df['S'].values,
            r = self.df['r'].values,
            T = self.df['T'].values,
            option_type = 'call'
        )

        return None