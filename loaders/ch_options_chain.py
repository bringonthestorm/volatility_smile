from datetime import datetime, timedelta
import pandas as pd

from loaders.ch_client import get_client
import loaders.ch_config as ch_cfg

def get_tardis_options_chain_1m(
    start_date: str, end_date: str, underlying_assets: str
) -> pd.DataFrame:

    client = get_client(
        host_ip = ch_cfg.staging['host_ip']['local'],
        database = ch_cfg.staging['database']
    )

    underlying_assets = [item.upper() for item in underlying_assets]

    query = f"""
        SELECT DISTINCT
            ts,
            underlying_asset,
            expiration, 
            symbol, 
            type, 
            strike_price, 
            open_interest,
            bid_price,
            ask_price,
            underlying_index, 
            underlying_price as forward_price, 
            mark_price, 
        FROM 
            options.vw_tardis_options_chain_1m 
        WHERE 
            underlying_asset in {underlying_assets}
            AND ts >= '{start_date}' 
            AND ts < '{end_date}'
    """

    tmp_df = client.query_dataframe(query)

    tmp_df['underlying_asset'] = tmp_df['underlying_asset'].str.lower()
    tmp_df['expiration'] = pd.to_datetime(tmp_df['expiration']/1000000, unit='s')

    tmp_df.sort_values(by=['underlying_asset', 'ts', 'expiration', 'strike_price'], inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    return tmp_df

def get_tardis_options_chain_1m_unique_timestamps(
    start_date: str, end_date: str, underlying_assets: list, interval_days: int = 10
) -> pd.DataFrame:
    client = get_client(
        host_ip = ch_cfg.staging['host_ip']['local'],
        database = ch_cfg.staging['database']
    )

    underlying_assets = [item.upper() for item in underlying_assets]
    all_timestamps = []  # List to hold all unique timestamps from each batch

    # Convert start and end dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    # Split the date range into intervals
    current_start_dt = start_dt
    while current_start_dt < end_dt:
        current_end_dt = min(current_start_dt + timedelta(days=interval_days), end_dt)
        
        # Format dates for the query
        query_start_date = current_start_dt.strftime('%Y-%m-%d %H:%M:%S')
        query_end_date = current_end_dt.strftime('%Y-%m-%d %H:%M:%S')

        query = f"""
            SELECT DISTINCT
                ts
            FROM 
                options.vw_tardis_options_chain_1m 
            WHERE 
                underlying_asset in {underlying_assets}
                AND ts >= '{query_start_date}'
                AND ts < '{query_end_date}'
        """

        tmp_df = client.query_dataframe(query)

        if not tmp_df.empty:
            tmp_df.sort_values(by=['ts'], inplace=True)
            tmp_df.reset_index(drop=True, inplace=True)

            all_timestamps.append(tmp_df)

        # Move to the next interval
        current_start_dt = current_end_dt

    # Concatenate all batches into a single DataFrame and remove duplicates
    if all_timestamps:
        final_df = pd.concat(all_timestamps, ignore_index=True).drop_duplicates().sort_values(by=['ts']).reset_index(drop=True)
    else:
        # Return an empty dataframe if no timestamps were found
        final_df = pd.DataFrame(columns=['ts'])

    return final_df

def get_tardis_options_chain_1m_from_timestamps(
    underlying_assets: str, timestamps: list[str], batch_size: int = 100
) -> pd.DataFrame:

    client = get_client(
        host_ip = ch_cfg.staging['host_ip']['local'],
        database = ch_cfg.staging['database']
    )

    underlying_assets = [item.upper() for item in underlying_assets]
    all_dfs = []  # List to hold dataframes from each batch

    # Split timestamps into batches
    for i in range(0, len(timestamps), batch_size):
        batch_timestamps = timestamps[i:i+batch_size]

        query = f"""
            SELECT DISTINCT
                ts,
                underlying_asset,
                expiration, 
                symbol, 
                type, 
                strike_price, 
                open_interest,
                bid_price,
                ask_price,
                underlying_index, 
                underlying_price as forward_price, 
                mark_price
            FROM 
                options.vw_tardis_options_chain_1m 
            WHERE 
                underlying_asset in {underlying_assets}
                AND ts in {batch_timestamps}
        """

        print(batch_timestamps)

        tmp_df = client.query_dataframe(query)

        if not tmp_df.empty:
            tmp_df['underlying_asset'] = tmp_df['underlying_asset'].str.lower()
            tmp_df['expiration'] = pd.to_datetime(tmp_df['expiration']/1000000, unit='s')

            tmp_df.sort_values(by=['underlying_asset', 'ts', 'expiration', 'strike_price'], inplace=True)
            tmp_df.reset_index(drop=True, inplace=True)

            all_dfs.append(tmp_df)

    # Concatenate all batches into a single DataFrame
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        # Return an empty dataframe with the same columns if no data was retrieved
        final_df = pd.DataFrame(columns=['ts', 'underlying_asset', 'expiration', 'symbol', 'type', 
                                         'strike_price', 'open_interest', 'bid_price', 'ask_price', 
                                         'underlying_index', 'forward_price', 'mark_price'])

    return final_df