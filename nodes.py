from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from pricing import (
    black_scholes_option_delta, 
    black_scholes_option_pricing
)

from volatility import StrikeVolatilityModelling

class NodesDataframe:

    def __init__(self, df_options: pd.DataFrame, extrapolation_type: int):

        self.nodes_delta = np.array([0.99, 0.97, 0.93, 0.85, 0.70, 0.50, 0.30, 0.15, 0.07, 0.03, 0.01])

        self.df_options = df_options
        self.extrapolation_type = extrapolation_type # volatility curve - tail extrapolation type
        self.df_nodes = self.run_combinations()

    def run_combinations(self) -> None:

        dataframes_list = []

        for _, row in self.df_options.drop_duplicates(['current_date', 'token', 'T']).iterrows():

            token = row['token']
            current_date = row['current_date']
            expiration_date = row['expiration_date']

            print(f"token: {token} | current date: {current_date} | expiration date: {expiration_date}")

            self.T = row['T']
            self.r = row['r']
            self.S = row['S']
            F = row['F']

            # OBSERVED DATA
            # ----------------------------------------------------- #
            df_options_tmp = self.df_options.loc[
                (self.df_options['token'] == token)
                &
                (self.df_options['current_date'] == current_date)
                &
                (self.df_options['expiration_date'] == expiration_date)
            ].copy(deep=True)

            df_options_tmp.sort_values(by='K', inplace=True)

            observed_delta_values = df_options_tmp['delta'].values
            observed_strike_values = df_options_tmp['K'].values
            observed_iv_values = df_options_tmp['iv'].values
            
            # we can observer the option price
            # some are puts other are calls
            # this computes the call price for all observations
            observed_call_price_values = [
                black_scholes_option_pricing(
                    observed_iv_values[i], observed_strike_values[i], self.S, self.r, self.T, option_type = 'call'
                ) for i in range(len(observed_iv_values))
            ]

            # Fit a CUBIC SPLINE to OBSERVED (STRIKE; IV)
            # ----------------------------------------------------- #
            self.cspline_strike_iv = CubicSpline(x = observed_strike_values, y = observed_iv_values)

            # NODES
            # For DELTAS -> compute the STRIKE, IV and Call Price
            # ----------------------------------------------------- #
            nodes_delta_values = self.nodes_delta
            nodes_strike_values = self._add_strikes_for_deltas(observed_delta_values, observed_strike_values)
            nodes_iv_values = np.array([float(self.cspline_strike_iv(K)) for K in nodes_strike_values])

            # Find indices of non-nan values
            nodes_non_nan_indices = np.where(~np.isnan(nodes_iv_values))[0]

            nodes_delta_values = nodes_delta_values[nodes_non_nan_indices]
            nodes_strike_values = nodes_strike_values[nodes_non_nan_indices]
            nodes_iv_values = nodes_iv_values[nodes_non_nan_indices]

            # If duplicate strike values exist, remove them
            unique_nodes_strike_values, indices = np.unique(nodes_strike_values, return_index=True)

            # If duplicates are found, keep only the unique values in all arrays
            if len(nodes_strike_values) != len(unique_nodes_strike_values):
                
                print(f"""BEFORE
                    - delta: {nodes_delta_values}
                    - strike: {nodes_strike_values}
                    - iv: {nodes_iv_values}
                """)

                # Filter the arrays based on the indices of unique values
                nodes_strike_values = nodes_strike_values[indices]
                nodes_iv_values = nodes_iv_values[indices]
                nodes_delta_values = nodes_delta_values[indices]

                print(f"""AFTER
                    - delta: {nodes_delta_values}
                    - strike: {nodes_strike_values}
                    - iv: {nodes_iv_values}
                """)


            svm = StrikeVolatilityModelling(
                strike_values = nodes_strike_values,
                volatility_values = nodes_iv_values,
                spot_price = self.S,
                extrapolation_type = self.extrapolation_type
            )

            if self.extrapolation_type in [1, 2]:
                nodes_strike_values = svm.strike_values
                nodes_iv_values = svm.volatility_values

                nodes_delta_values = np.concatenate((
                    nodes_delta_values, self._get_delta_values(svm.extrapolated_strikes_rt, svm.extrapolated_vols_rt)
                ))

            nodes_call_price_values = np.array([
                black_scholes_option_pricing(
                    nodes_iv_values[i], nodes_strike_values[i], self.S, self.r, self.T, option_type = 'call'
                ) for i in range(len(nodes_iv_values))
            ])

            # SPLINE MODEL
            # ----------------------------------------------------- #
            func_callable = svm.get_iv_for_strike
            
            # Add data to dictionary
            # ----------------------------------------------------- #

            data = {
                'token': token,
                'current_date': current_date,
                'expiration_date': expiration_date,
                'T': self.T,
                'r': self.r,
                'S': self.S,
                'F': F,
                'observed_delta': [observed_delta_values],
                'observed_K': [observed_strike_values],
                'observed_iv': [observed_iv_values],
                'observed_C': [observed_call_price_values],
                'extrapolation_type': self.extrapolation_type,
                'nodes_delta': [nodes_delta_values],
                'nodes_K': [nodes_strike_values],
                'nodes_iv': [nodes_iv_values],
                'nodes_C': [nodes_call_price_values],
                'func_callable': func_callable
            }

            dataframes_list.append(pd.DataFrame(data=data)) 

        return pd.concat(dataframes_list).reset_index(drop=True)


    def _get_delta_values(self, strike_values: np.ndarray, volatility_values: np.ndarray):

        return np.array([
            black_scholes_option_delta(
                volatility_values[i], strike_values[i], self.S, self.r, self.T, option_type='call'
            ) for i in range(len(strike_values))
        ])


    def _node_strike_binary_search(
        self, delta_node: float, low_strike: float, upper_strike: float, tolerance: float=3e-2, max_iterations: int=100
    ) -> (float, None):

        count = 0

        while low_strike <= upper_strike:
            count += 1
            # compute current strike value with the upper_strike and lower_strike
            curr_strike = low_strike + (upper_strike - low_strike) / 2

            # compute implied volatility for the curr_strike with the previously 
            # calibrated cubic spline function
            iv = self.cspline_strike_iv(curr_strike)

            curr_delta = black_scholes_option_delta(iv, curr_strike, self.S, self.r, self.T, option_type='call')

            if abs(curr_delta - delta_node) < tolerance:
                return curr_strike
            elif curr_delta > delta_node:
                low_strike = curr_strike
            else:
                upper_strike = curr_strike
            if count > max_iterations:
                return curr_strike

        return np.nan

    def _add_strikes_for_deltas(self, delta_values: list[float], strike_values: list[float]) -> np.ndarray:

        # Construct the dictionary using zip
        delta_strike_pairs = dict(zip(delta_values, strike_values))
        # Get sorted deltas from "delta_strike_pairs" dictionary
        observed_sorted_deltas = list(sorted(delta_strike_pairs))

        # print(delta_strike_pairs)

        nodeDelta_strike_pairs = {}

        for delta_node in self.nodes_delta:
            lower_strike = None
            upper_strike = None

            # as the delta increases, the strike price decreases
            for observerd_delta in observed_sorted_deltas:
                if delta_node > observerd_delta:
                    upper_strike = delta_strike_pairs.get(observerd_delta)
                elif delta_node < observerd_delta:
                    lower_strike = delta_strike_pairs.get(observerd_delta)
                    break
            
            if (not lower_strike) or (not upper_strike):
                nodeDelta_strike_pairs[delta_node] = np.nan
            else:
                nodeDelta_strike_pairs[delta_node] = self._node_strike_binary_search(delta_node, lower_strike, upper_strike)

        return np.array(list(nodeDelta_strike_pairs.values()))