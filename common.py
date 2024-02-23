from bisect import bisect_left
from datetime import datetime
from typing import Union
import pandas as pd
import numpy as np


def filter_dataframe_by_date_token_extrapolation(
    df: pd.DataFrame, token: str, date: str, extrapolation_type: int
) -> pd.DataFrame:
    """ Filter data based on date and token """
    return df.loc[(df['current_date'] == date) & (df['token'] == token) & (df['extrapolation_type'] == extrapolation_type)]

def filter_dataframe_by_token_current_date_expiration_date_extrapolation(
    df: pd.DataFrame, token: str, current_date: str, expiration_date: str, extrapolation_type: int
) -> pd.DataFrame:
    """
    Filter the DataFrame based on given criteria.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter.
    - token (str): The token to filter by.
    - expiration_date (str): The expiration date to filter by.
    - current_date (str): The current date to filter by.
    - extrapolation_type (int): The extrapolation type to filter by.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    return df.loc[
        (df['token'] == token) &
        (df['current_date'] == current_date) &
        (df['expiration_date'] == expiration_date) &
        (df['extrapolation_type'] == extrapolation_type)
    ]

def format_dates(dates: list) -> list:
    """ Convert dates to a readable format """
    return [datetime.utcfromtimestamp(d.astype('datetime64[s]').astype(int)).strftime('%Y-%m-%d %H:%M:%S') for d in dates]

def convert_str_to_datetime(date_string: str) -> datetime:
    """
    Converts a string in the format 'YYYY-MM-DD HH:MM:SS' to a datetime object.

    Args:
    date_string (str): The date string to convert.

    Returns:
    datetime: A datetime object representing the input string.
    """
    try:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        print(f"Error converting '{date_string}' to datetime: {e}")
        return None

def convert_str_to_datetime64(date_string: str) -> np.datetime64:
    """
    Converts a string in the format 'YYYY-MM-DD HH:MM:SS' to a numpy.datetime64 object.

    Args:
    date_string (str): The date string to convert.

    Returns:
    numpy.datetime64: A numpy.datetime64 object representing the input string.
    """
    try:
        return np.datetime64(date_string)
    except ValueError as e:
        print(f"Error converting '{date_string}' to numpy.datetime64: {e}")
        return None


def find_start_times(datetime_list, period):
    """
    Finds the first datetime at the start of the specified period (day, hour, or minute).

    Args:
    datetime_list (list): A list of numpy.datetime64 objects.
    period (str): 'day', 'hour', or 'minute' to specify the period.

    Returns:
    list: A list of formatted date strings representing the start of the specified period.
    """

    # Convert to Python datetime objects for easier manipulation
    datetime_list = [datetime.utcfromtimestamp(d.astype('O')/1e9) for d in datetime_list]

    # Initialize an empty set to keep track of unique periods
    unique_periods = set()
    start_times = []

    for dt in datetime_list:
        if period == 'day':
            period_start = datetime(dt.year, dt.month, dt.day)
        elif period == 'hour':
            period_start = datetime(dt.year, dt.month, dt.day, dt.hour)
        elif period == 'minute':
            period_start = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        else:
            raise ValueError("Invalid period. Choose 'day', 'hour', or 'minute'.")

        if period_start not in unique_periods:
            unique_periods.add(period_start)
            start_times.append(dt)

    # Format the dates
    formatted_dates = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in start_times]

    return formatted_dates


def is_date_within_interval(dates: list[pd.Timestamp], date: pd.Timestamp) -> bool:
    """

    Returns:
    - bool: True if date is within the interval, False otherwise.
    """

    if isinstance(dates[0], pd.Timestamp) and isinstance(date, pd.Timestamp):
        dates = sorted(dates)

        print(date, dates)
        return (date >= dates[0]) and (date <= dates[-1])
    else:
        raise ValueError("dates type != pd.Timestamp")


def find_strict_nearest_date_bounds(dates: list[pd.Timestamp], input_date: pd.Timestamp) -> tuple:
    """
    Find the closest lower and upper bound dates to the input date, ensuring bounds are not equal to the input date.

    Parameters:
    - input_date (pd.Timestamp): pd.Timestamp('YYYY-MM-DD HH:MM:SS').
    - dates (list[pd.Timestamp]): A list of dates in pd.Timestamp format.

    Returns:
    - tuple: A tuple containing the closest lower and upper bound dates as strings. 
             Returns (None, None) if no bounds are found.
    """

    dates = sorted(dates)
    index = bisect_left(dates, input_date)
    
    lower_bound = dates[index - 1] if index > 0 else None
    upper_bound = dates[index] if index < len(dates) else None

    # Adjust bounds if they are equal to the input date
    if lower_bound == input_date:
        lower_bound = dates[index - 2] if index - 1 > 0 else None
    if upper_bound == input_date:
        upper_bound = dates[index + 1] if index + 1 < len(dates) else None
    
    return (lower_bound, upper_bound)

# Auxiliary Functions
# -------------------
def calculate_time_to_maturity(
    current_date: Union[str, pd.Timestamp, np.datetime64], 
    expiration_date: Union[str, pd.Timestamp, np.datetime64]
) -> float:
    """
    Calculate the time to maturity in years.

    Parameters:
    - current_date (str; pd.Timestamp): 
        Current date string in the format 'YYYY-MM-DD HH:MM:SS' or pd.Timestamp.
    - expiration_date (str; pd.Timestamp): 
        Expiration date string in the format 'YYYY-MM-DD HH:MM:SS' or pd.Timestamp.

    Returns:
    - float: Time to maturity in years.
    """

    # Convert string to datetime
    if isinstance(current_date, str):
        current_date = convert_str_to_datetime(current_date)

    if isinstance(expiration_date, str):
        expiration_date = convert_str_to_datetime(expiration_date)

    # Calculate time difference in seconds and convert to years
    time_diff_seconds = (expiration_date - current_date).total_seconds()
    time_to_maturity_years = time_diff_seconds / (365.25 * 24 * 3600)  # Considering leap years

    return time_to_maturity_years