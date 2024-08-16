import numpy as np
import pandas as pd

from typing import Tuple


def get_common_timestamps(data: pd.DataFrame, symbols: Tuple[str, str]) -> pd.DataFrame:
    """
    Finds and retains only the common timestamps across all given symbols in the dataset,
    and filters the dataset to include only the specified symbols.
    """
    common_timestamps = set(data[data['ticker'] == symbols[0]].dropna().index)
    for symbol in symbols[1:]:
        common_timestamps &= set(data[data['ticker'] == symbol].dropna().index)

    common_timestamps = sorted(common_timestamps)
    filtered_dataset = data[data['ticker'].isin(symbols)]
    return filtered_dataset.loc[common_timestamps]


def filter_current_date(data: pd.DataFrame, current_date: str) -> pd.DataFrame:
    """
    Filters dataset between 10:00 AM and 19:00 PM UTC on the specified date.
    """
    data.index = pd.to_datetime(data.index)
    start_datetime = pd.to_datetime(current_date).tz_localize('UTC') + pd.Timedelta(hours=10, minutes=0)
    end_datetime = pd.to_datetime(current_date).tz_localize('UTC') + pd.Timedelta(hours=21, minutes=0)
    return data[(data.index >= start_datetime) & (data.index <= end_datetime)]


def get_data_and_preprocess(csv_path: str, symbols: Tuple[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads a CSV file, filters data for common timestamps across specified symbols, and retrieves closing prices and trading volumes.
    """
    data = pd.read_csv(csv_path, sep=';', index_col='times', parse_dates=True)
    common_data = get_common_timestamps(data, symbols)
    prices = common_data.pivot(columns='ticker', values='close').dropna()
    volumes = common_data.pivot(columns='ticker', values='volume').dropna()
    return common_data, prices, volumes
