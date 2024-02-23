import pandas as pd

from loaders.ch_client import get_client
import loaders.ch_config as ch_cfg

def get_realized_volatility_1m(
    start_date: str, end_date: str, instrument_ids: str, rolling_window: str = '6_month', period: str='day'
) -> pd.DataFrame:

    client = get_client(
        host_ip = ch_cfg.test['host_ip']['local'],
        database = ch_cfg.test['database']
    )

    query = f"""
        SELECT DISTINCT
            end_ts as ts,
            instrument_id,
            rolling_window,
            resample_period,
            annualized_realized_volatility
        FROM
            test.s3_public_trades_volatility
        WHERE
            resample_period == '1_minute'
            AND rolling_window == '{rolling_window}'
            AND ts >= 1000 * toUnixTimestamp('{start_date}')
            AND ts < 1000 * toUnixTimestamp('{end_date}')
            AND instrument_id in {instrument_ids}
    """

    tmp_df = client.query_dataframe(query)
    tmp_df.sort_values(by=['instrument_id', 'ts'], inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)
    tmp_df['ts'] = pd.to_datetime(arg=tmp_df['ts'].astype("int64"), unit='ms')

    # Generate a complete datetime index from start_date to end_date
    if period == 'day':
        frequency = 'D'
    elif period == 'hour':
        frequency = 'h'
    elif period == 'minute':
        frequency = 'min'
        
    complete_index = pd.date_range(start=start_date, end=end_date, freq=frequency)

    # Create a DataFrame from the complete datetime index
    complete_df = pd.DataFrame(complete_index, columns=['ts'])
    complete_df['key'] = 0

    # Create a DataFrame of unique instrument_id
    assets_df = pd.DataFrame(tmp_df['instrument_id'].unique(), columns=['instrument_id'])
    assets_df['key'] = 0

    # Create a Cartesian product of complete datetime index and unique assets
    all_combinations = pd.merge(complete_df, assets_df, on='key').drop(columns='key')

    # Merge with the original data
    tmp_df = pd.merge(all_combinations, tmp_df, on=['instrument_id', 'ts'], how='left')

    # Forward fill and backfill the missing values
    tmp_df['annualized_realized_volatility'] = tmp_df.groupby('instrument_id')['annualized_realized_volatility'].ffill().bfill()
    tmp_df['rolling_window'] = tmp_df.groupby('instrument_id')['rolling_window'].ffill().bfill()
    tmp_df['resample_period'] = tmp_df.groupby('instrument_id')['resample_period'].ffill().bfill()

    tmp_df['exchange'] = tmp_df['instrument_id'].str.split('_').str[0]
    tmp_df['base_currency'] = tmp_df['instrument_id'].str.split('_').str[1]
    tmp_df['quote_currency'] = tmp_df['instrument_id'].str.split('_').str[2]
    tmp_df['type'] = tmp_df['instrument_id'].str.split('_').str[3]

    return tmp_df

def get_realized_volatility(
    start_date: str, end_date: str, instrument_ids: str, 
    resample_period: str = '1_minute', rolling_window: str = '6_month'
) -> pd.DataFrame:

    client = get_client(
        host_ip = ch_cfg.test['host_ip']['local'],
        database = ch_cfg.test['database']
    )

    query = f"""
        SELECT DISTINCT
            end_ts as ts,
            instrument_id,
            rolling_window,
            resample_period,
            annualized_realized_volatility
        FROM
            test.s3_public_trades_volatility
        WHERE
            resample_period == '{resample_period}'
            AND rolling_window == '{rolling_window}'
            AND ts >= 1000 * toUnixTimestamp('{start_date}')
            AND ts < 1000 * toUnixTimestamp('{end_date}')
            AND instrument_id in {instrument_ids}
    """

    tmp_df = client.query_dataframe(query)

    tmp_df.sort_values(by=['instrument_id', 'ts'], inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    tmp_df['ts'] = pd.to_datetime(
        arg=tmp_df['ts'].astype("int64"), unit='ms'
    )

    tmp_df['exchange'] = tmp_df['instrument_id'].str.split('_').str[0]
    tmp_df['base_currency'] = tmp_df['instrument_id'].str.split('_').str[1]
    tmp_df['quote_currency'] = tmp_df['instrument_id'].str.split('_').str[2]
    tmp_df['type'] = tmp_df['instrument_id'].str.split('_').str[3]

    return tmp_df