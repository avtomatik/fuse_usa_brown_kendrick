#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:25:31 2023

@author: green-machine
"""


import pandas as pd
from core.backend import enlist_series_ids, stockpile
from core.classes import Dataset


def get_fused_usa_brown_kendrick() -> pd.DataFrame:
    """
    Fetch Data from:
        <reference_ru_brown_m_0597_088.pdf>, Page 193 &
        Out of J.W. Kendrick Data & Table 2. of <reference_ru_brown_m_0597_088.pdf>
    FN:Murray George Brown
    ORG:University at Buffalo;Economics
    TITLE:Professor Emeritus, Retired
    EMAIL;PREF;INTERNET:mbrown@buffalo.edu

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """

    SERIES_IDS = enlist_series_ids(
        map(lambda _: f'brown_{hex(_)}', range(6)),
        Dataset.USA_BROWN
    )
    df_b = stockpile(SERIES_IDS)

    SERIES_IDS = ['KTA03S07', 'KTA03S08', 'KTA10S08', 'KTA15S07', 'KTA15S08']
    df_k = stockpile(
        enlist_series_ids(SERIES_IDS, Dataset.USA_KENDRICK)
    ).truncate(before=1889).truncate(after=1954)

    df = pd.concat(
        [
            # =================================================================
            # Omit Two Last Rows
            # =================================================================
            df_k[~df_k.index.duplicated(keep='first')],
            # =================================================================
            # Первая аппроксимация рядов загрузки мощностей, полученная с помощью метода Уортонской школы
            # =================================================================
            df_b.loc[:, ['brown_0x4']].truncate(after=1953)
        ],
        axis=1,
        sort=True
    )
    df = df.assign(
        brown_0x0=df.iloc[:, 0].sub(df.iloc[:, 1]),
        brown_0x1=df.iloc[:, 3].add(df.iloc[:, 4]),
        brown_0x2=df.iloc[:, [3, 4]].sum(axis=1).rolling(
            2).mean().mul(df.iloc[:, 5]).div(100),
        brown_0x3=df.iloc[:, 2]
    )
    return pd.concat(
        [
            df.iloc[:, -4:].dropna(axis=0),
            # =================================================================
            # M.G. Brown Numbers Not Found in J.W. Kendrick For Years Starting From 1954 Inclusive
            # =================================================================
            df_b.iloc[:, range(4)].truncate(before=1954)
        ]
    ).round()


if __name__ == '__main__':

    print(get_fused_usa_brown_kendrick())
