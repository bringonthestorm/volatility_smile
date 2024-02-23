from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def iterate_over_current_dates(
    *args, token: str, current_dates: np.ndarray, gap_between_current_dates: timedelta, func: callable
) -> None:

    # Set the minimum time difference
    min_time_difference = timedelta(minutes=gap_between_current_dates)

    # Convert current_date to datetime objects
    datetime_current_dates = [datetime.strptime(str(current_date), "%Y-%m-%d %H:%M:%S") for current_date in current_dates]

    datetime_current_date = 0

    # Loop through datetime current_dates
    for i in range(len(datetime_current_dates)):

        if i > 0:
            dts_diff = (datetime_current_dates[i] - datetime_current_date)
        else:
            dts_diff = False
            
        # Check if it's the first current_date or if 10 minutes have passed
        if i == 0 or (dts_diff) >= min_time_difference:

            func(*args, token=token, current_date=current_dates[i])

            datetime_current_date = datetime_current_dates[i]


def volatility_smile_fixed_T(*args):

    df_nodes = args[0].copy(deep=True)

    # Get unique EXPIRATION DATE values
    expiration_dates = list(df_nodes['expiration_date'].unique())
    # Get TOKEN name
    token = df_nodes['token'].values[0]
    # Get CURRENT DATE value
    current_date = df_nodes['current_date'].values[0]
    # Get CURRENT SPOT PRICE
    spot_price = df_nodes['S'].values[0]

    # Determine the number of rows and columns for subplots
    num_plots = len(expiration_dates)
    num_cols = 3  # You can adjust the number of columns as needed
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 6*num_rows))

    # Flatten the axes array if it's more than 1D
    axes = axes.flatten()

    # Loop through unique T values and create subplots
    for i, expiration_date in enumerate(expiration_dates):

        df_nodes_tmp = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date)
        ]
            
        expiration_date = df_nodes_tmp['expiration_date'].values[0]
        expiration_date = np.datetime_as_string(expiration_date, unit='s')
        expiration_date = datetime.strptime(expiration_date, "%Y-%m-%dT%H:%M:%S")
        expiration_date = expiration_date.strftime('%Y-%m-%d %H:%M:%S')

        # Select the appropriate subplot
        ax = axes[i]

        # Scatter plot
        ax.scatter(
            x = np.concatenate(df_nodes_tmp[f"observed_{args[1]}"].values),
            y = np.concatenate(df_nodes_tmp['observed_iv'].values),
            c='red', label='Observed'
        )


        # Line plot
        ax.plot(
            np.concatenate(df_nodes_tmp[f"nodes_{args[1]}"].values),
            np.concatenate(df_nodes_tmp['nodes_iv'].values),
            c='blue', label='Train', marker='.', mfc='g', mec='g', ls='-', ms=12
        )

        # Add legend
        ax.legend()

        if args[1] == 'K':
            x_label = 'Strike Price'
        elif args[1] == 'delta':
            x_label = 'Delta'
        else:
            raise ValueError("x-axis can only be equal to 'K' or 'delta'.")

        # Add axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Expiration Date: {expiration_date}')

        # ax.set_xlim(0.5 * spot_price, 3 * spot_price)

    # Add a title for the entire plot
    plt.suptitle(f'Token: {token.upper()} | Current Date: {current_date}', fontsize=14, fontweight='bold')
    # Adjust layout to prevent overlapping and add vertical space
    plt.subplots_adjust(top=0.94, hspace=0.34)

    # Show the plots
    plt.show()

    return None


def volatility_smile_fixed_T_sample_strike_values(*args):

    df_nodes = args[0].copy(deep=True)

    display(df_nodes['token'].values[0])

    # Get unique EXPIRATION DATE values
    expiration_dates = list(df_nodes['expiration_date'].unique())
    # Get TOKEN name
    token = df_nodes['token'].values[0]
    # Get CURRENT DATE value
    current_date = df_nodes['current_date'].values[0]
    # Get CURRENT SPOT PRICE
    spot_price = df_nodes['S'].values[0]
        
    # Compute range of STRIKE PRICE values
    strike_price_values = np.linspace(
        0.5 * spot_price, 5 * spot_price, 1000
    )


    print(f"""
        - token: {token}
        - current_date : {current_date}
        - spot price: {spot_price}
    """)

    # Determine the number of rows and columns for subplots
    num_plots = len(expiration_dates)
    num_cols = 3  # You can adjust the number of columns as needed
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 6*num_rows))

    # Flatten the axes array if it's more than 1D
    axes = axes.flatten()

    # Loop through unique T values and create subplots
    for i, expiration_date in enumerate(expiration_dates):

        func = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date), 'func_callable'
        ].values[0]

        iv_values = np.array([func(strike_price) for strike_price in strike_price_values])

        # Select the appropriate subplot
        ax = axes[i]

        # Line plot
        ax.plot(strike_price_values, iv_values, c='blue', label = 'sample data')

        # Observed Data Points
        observed_strike_values = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date), 'observed_K'
        ].values[0]

        observed_iv_values = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date), 'observed_iv'
        ].values[0]

        ax.scatter(
            x = observed_strike_values, y = observed_iv_values, color='r', label='observed data'
        )

        # Nodes Data Points
        node_strike_values = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date), 'nodes_K'
        ].values[0]

        node_iv_values = df_nodes.loc[
            (df_nodes['expiration_date'] == expiration_date), 'nodes_iv'
        ].values[0]

        ax.scatter(
            x = node_strike_values, y = node_iv_values, color='g', label='node data'
        )

        # Add axis labels
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Expiration Date: {expiration_date}')

        ax.legend()

    # Add a title for the entire plot
    plt.suptitle(f'Token: {token.upper()} | Current Date: {current_date}', fontsize=14, fontweight='bold')
    # Adjust layout to prevent overlapping and add vertical space
    plt.subplots_adjust(top=0.94, hspace=0.34)

    # Show the plots
    plt.show()

    return None