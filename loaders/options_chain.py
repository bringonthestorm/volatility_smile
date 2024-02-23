import pandas as pd

from loaders.source_options import SourceOptionsDataProcessor
from loaders.target_options import TargetOptionsDataframe

def load_options_chain_dataframe(start_date: str, end_date: str, tokens: dict, symbols: dict, instrument_ids: dict, period: str) -> pd.DataFrame:

    df_source_options = SourceOptionsDataProcessor(
        start_date, end_date, tokens['source'], symbols['source'], instrument_ids['source'], period=period
    ).df.copy(deep=True)

    df_target_options = TargetOptionsDataframe(
        start_date, end_date, tokens['source'][0], tokens['target'], symbols['target'], instrument_ids['target'], df_source_options, period=period
    ).df.copy(deep=True)

    df_options = pd.concat([df_source_options, df_target_options]).reset_index(drop=True)

    return df_options