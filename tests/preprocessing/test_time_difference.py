import numpy as np
import pandas as pd
import pytest

from datetime import date, datetime

from src.preprocessing.time_difference import TimeDifference, Birth


class TestTimeDifference(object):

    def test_dates2dayduration_valid(self):
        opened = datetime(2020, 5, 17)
        closed = datetime(2020, 5, 20)
        day_duration = TimeDifference().dates2dayduration(opened_date=opened,
                                                          closed_date=closed)
        assert day_duration == 3

    def test_dates2dayduration_valid_over_month(self):
        opened = datetime(2020, 5, 17)
        closed = datetime(2020, 6, 20)
        day_duration = TimeDifference().dates2dayduration(opened_date=opened,
                                                          closed_date=closed)
        assert day_duration == 34

    def test_dates2dayduration_closed_before_open(self):
        closed = datetime(2020, 5, 17)
        opened = datetime(2020, 6, 20)
        day_duration = TimeDifference().dates2dayduration(opened_date=opened,
                                                          closed_date=closed)
        print(day_duration)

        assert day_duration == 34

    def test_dates2dayduration_pandas(self):

        df = pd.DataFrame({
            "open": [datetime(2020, 5, 17), datetime(2022, 5, 17)],
            "closed": [datetime(2020, 5, 20), datetime(2022, 5, 17)],
            "day_duration_expected": [3, 0]
        })

        df['day_duration'] = df.apply(lambda row: TimeDifference().dates2dayduration(opened_date=row["open"],
                                                                                     closed_date=row["closed"]), axis=1)

        assert df['day_duration'].equals(df['day_duration_expected'])

    def test_dates2dayduration_pandas_without_closed(self):

        df = pd.DataFrame({
            "open": [datetime(2020, 5, 17), datetime(2022, 5, 17)],
            "closed": [datetime(2020, 5, 20), np.nan],
            "day_duration_expected": [3, np.nan]
        })

        df['day_duration'] = df.apply(lambda row: TimeDifference().dates2dayduration(opened_date=row["open"],
                                                                                     closed_date=row["closed"]), axis=1)

        assert df['day_duration'].equals(df['day_duration_expected'])


class TestBith(object):

    def test_date2age_valid(self):
        bithday = date(2020, 5, 17)
        age = Birth().date2age(born=bithday)
        print(age)

        assert age == 2

    def test_date2age_future(self):
        bithday = date(2023, 5, 17)
        age = Birth().date2age(born=bithday)
        print(age)

        assert age == -1

