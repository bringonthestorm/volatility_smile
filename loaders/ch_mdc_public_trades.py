import pandas as pd

from loaders.ch_client import get_client
import loaders.ch_config as ch_cfg

def get_ag_mdc_public_trades_ohlc_1m(start_date: str, end_date: str, symbols: str) -> pd.DataFrame:
    # Assuming get_client and ch_cfg are defined elsewhere
    client = get_client(
        host_ip = ch_cfg.test['host_ip']['local'],
        database = ch_cfg.test['database']
    )

    query = f"""
        SELECT DISTINCT
            ts,
            base as underlying_asset,
            vwap as spot_price
        FROM
            test.vw_ag_mdc_public_trades_ohlc_1m
        WHERE
            symbol in {symbols}
            AND ts >= 1000 * toUnixTimestamp('{start_date}')
            AND ts < 1000 * toUnixTimestamp('{end_date}')
    """

    tmp_df = client.query_dataframe(query)

    tmp_df.sort_values(by=['underlying_asset', 'ts'], inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    tmp_df['ts'] = pd.to_datetime(
        arg=tmp_df['ts'].astype("int64"), unit='ms'
    )

    # Generate a complete datetime index from start_date to end_date
    complete_index = pd.date_range(start=start_date, end=end_date, freq='T')

    # Create a DataFrame from the complete datetime index
    complete_df = pd.DataFrame(complete_index, columns=['ts'])
    complete_df['key'] = 0

    # Create a DataFrame of unique underlying assets
    assets_df = pd.DataFrame(tmp_df['underlying_asset'].unique(), columns=['underlying_asset'])
    assets_df['key'] = 0

    # Create a Cartesian product of complete datetime index and unique assets
    all_combinations = pd.merge(complete_df, assets_df, on='key').drop(columns='key')

    # Merge with the original data
    tmp_df = pd.merge(all_combinations, tmp_df, on=['underlying_asset', 'ts'], how='left')

    # Forward fill and backfill the missing values
    tmp_df['spot_price'] = tmp_df.groupby('underlying_asset')['spot_price'].ffill().bfill()

    # Check if any ts value is missing
    missing_ts = tmp_df[tmp_df['spot_price'].isnull()]
    print("Missing ts values:", len(missing_ts[['ts', 'underlying_asset']]))

    return tmp_df