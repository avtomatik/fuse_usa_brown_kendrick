#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:43:31 2023

@author: green-machine
"""


from functools import cache
from typing import Union

import pandas as pd
from core.classes import URL, Dataset, SeriesID
from pandas import DataFrame


def enlist_series_ids(series_ids: list[str], source: Union[Dataset, URL]) -> list[SeriesID]:
    return list(map(lambda _: SeriesID(_, source), series_ids))


def stockpile(series_ids: list[SeriesID]) -> DataFrame:
    """


    Parameters
    ----------
    series_ids : list[SeriesID]
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        ...                ...
        df.iloc[:, -1]     Values
        ================== =================================.

    """
    return pd.concat(
        map(
            lambda _: read_source(_).pipe(pull_by_series_id, _),
            series_ids
        ),
        axis=1,
        sort=True
    )


@cache
def read_source(series_id: SeriesID) -> DataFrame:
    """


    Parameters
    ----------
    series_id : SeriesID
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series IDs
        df.iloc[:, 1]      Values
        ================== =================================.

    """
    return pd.read_csv(**series_id.source.get_kwargs())


def pull_by_series_id(df: DataFrame, series_id: SeriesID) -> DataFrame:
    """


    Parameters
    ----------
    df : DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series IDs
        df.iloc[:, 1]      Values
        ================== =================================.
    series_id : SeriesID
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series
        ================== =================================.

    """
    assert df.shape[1] == 2
    return df[df.iloc[:, 0] == series_id.series_id].iloc[:, [1]].rename(
        columns={"value": series_id.series_id}
    )
