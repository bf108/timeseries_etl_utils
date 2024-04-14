import pytest

from datetime import datetime, date, timedelta
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from timeseries_etl_utils.utils import (
    get_zero_hrs_of_date,
    get_last_hrs_of_date,
    create_hourly_step_calendar,
    create_daily_step_calendar,
    read_sales_from_disk,
    create_full_sales_history,
    zero_small_and_negatives_sales,
    get_sales_n_days_prior,
    create_operation_flag_column,
    create_dom_column,
    create_hour_column,
    create_month_id_column,
    create_dow_column,
    n_week_forecast,
    create_week_id_column,
    fill_const_columns,
    drop_cos_sin_cols,
    create_year_column,
    create_date_only_column,
    convert_daily_hols_to_hours,
    create_n_days_empty_sales,
    append_n_days_of_empty_sales_on_historical_sales,
    create_perc_err_column,
    aggregate_daily_sales,
    count_positive_sales_days_in_last_n_days,
    create_sales_cap_column,
    calculate_hourly_avg_contribution,
    left_anti_join,
    create_optimal_hol_sf_column,
    create_label_optimal_hol_sf_column,
    create_error_column,
)


def test_convert_daily_hols_to_hours() -> None:
    df = pd.DataFrame(
        {
            "date": [date(2021, 12, 25), date(2021, 12, 26), date(2021, 12, 28)],
        },
    )
    df["date"] = pd.to_datetime(df["date"])
    df_output = convert_daily_hols_to_hours(df)
    assert df_output.shape[0] == 3 * 24


@pytest.fixture
def dir_testing_data() -> Path:
    return Path(__file__).parent.parent / "testing_data"


def test_read_sales_from_disk(dir_testing_data) -> None:
    f_path = dir_testing_data / "med.csv"
    df_med_sales = read_sales_from_disk(f_path)

    assert list(df_med_sales.columns) == [
        "unique_id",
        "ds",
        "y",
        "operational_flag",
    ]
    assert df_med_sales["ds"].dtype == np.dtype("<M8[ns]")
    assert df_med_sales["y"].dtype == np.dtype("float64")


@pytest.fixture
def df_med_sales(dir_testing_data):
    f_path = dir_testing_data / "med.csv"
    return read_sales_from_disk(f_path)


@pytest.mark.parametrize(
    ("input", "expected"),
    (
            (datetime(2023, 10, 15, 14, 37, 1), datetime(2023, 10, 15, 0, 0, 0)),
            (datetime(2021, 1, 1, 1, 30, 59), datetime(2021, 1, 1, 0, 0, 0)),
    ),
)
def test_get_zero_hrs_of_date(input: datetime, expected: datetime) -> None:
    assert get_zero_hrs_of_date(input) == expected


@pytest.mark.parametrize(
    ("input", "expected"),
    (
            (datetime(2023, 10, 15, 14, 37, 1), datetime(2023, 10, 15, 23, 0, 0)),
            (datetime(2021, 1, 1, 1, 30, 59), datetime(2021, 1, 1, 23, 0, 0)),
    ),
)
def test_get_last_hrs_of_date(input: datetime, expected: datetime) -> None:
    assert get_last_hrs_of_date(input) == expected


def test_create_hourly_step_calendar() -> None:
    start_dt = date(2023, 10, 1)
    end_dt = datetime(2023, 10, 7)
    df = create_hourly_step_calendar(start_dt, end_dt)

    # Assert df correct length
    assert df.shape[0] == 7 * 24
    assert df.columns == ["ds"]
    assert df.loc[0, "ds"] == datetime(2023, 10, 1, 0, 0, 0)
    assert df.iloc[-1]["ds"] == datetime(2023, 10, 7, 23, 0, 0)


def test_create_hourly_step_calendar_1day() -> None:
    start_dt = date(2022, 12, 31)
    end_dt = datetime(2022, 12, 31)
    df = create_hourly_step_calendar(start_dt, end_dt)

    # Assert df correct length
    assert df.shape[0] == 24
    assert df.columns == ["ds"]
    assert df.loc[0, "ds"] == datetime(2022, 12, 31, 0, 0, 0)
    assert df.iloc[-1]["ds"] == datetime(2022, 12, 31, 23, 0, 0)


def test_create_daily_step_calendar() -> None:
    start_dt = date(2022, 12, 25)
    end_dt = datetime(2023, 1, 1)
    df = create_daily_step_calendar(start_dt, end_dt)

    # Assert df correct length
    assert df.shape[0] == 8
    assert df.columns == ["ds"]
    assert df.loc[0, "ds"] == datetime(2022, 12, 25)
    assert df.iloc[-1]["ds"] == datetime(2023, 1, 1)


def test_zero_missing_hourly_sales(dir_testing_data: Path) -> None:
    start_dt = date(2022, 1, 1)
    end_dt = datetime(2022, 12, 31)
    f_path = dir_testing_data / "med.csv"
    df = read_sales_from_disk(f_path)
    df_comb = create_full_sales_history(start_dt, end_dt, df)

    assert df_comb.shape[0] == 365 * 24
    assert df_comb.shape[1] == 4
    assert sorted(list(df_comb.columns)) == sorted(
        ["unique_id", "ds", "operational_flag", "y"]
    )


def test_zero_missing_hourly_sales_leap_year(df_med_sales) -> None:
    start_dt = date(2020, 1, 1)
    end_dt = datetime(2020, 12, 31)
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)

    # leap year
    assert df_comb.shape[0] == 366 * 24
    assert df_comb.shape[1] == 4
    assert sorted(list(df_comb.columns)) == sorted(
        ["unique_id", "ds", "operational_flag", "y"]
    )


def test_fillna_create_full_sales_history(df_med_sales) -> None:
    start_dt = date(2020, 12, 31)
    end_dt = datetime(2020, 12, 31)
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)

    # leap year
    assert df_comb.shape[0] == 24
    assert df_comb.shape[1] == 4
    assert sorted(list(df_comb.columns)) == sorted(
        ["unique_id", "ds", "operational_flag", "y"]
    )
    assert df_comb["y"].sum() == 4298.78
    assert df_comb["y"].isna().sum() == 0
    assert df_comb["unique_id"].isna().sum() == 0


def test_zero_small_and_negatives_sales() -> None:
    df = pd.DataFrame({"y": [-10.0, -1.0, 0.0, 0.5, 0.9, 1.0, 10.0]})
    df_output = zero_small_and_negatives_sales(df)
    assert df_output[df_output["y_adj"] < 1].shape[0] == 5


@pytest.mark.parametrize(
    ("input_datetime", "shift_days"),
    (
            (datetime(2022, 1, 1, 12), 7),
            (datetime(2022, 2, 1, 12), 14),
            (datetime(2022, 6, 6, 0), 28),
            (datetime(2022, 10, 7, 1), 35),
    ),
)
def test_get_sales_n_days_prior(
        df_med_sales: pd.DataFrame, input_datetime: datetime, shift_days: int
) -> None:
    sales_exp = df_med_sales[(df_med_sales["ds"] == input_datetime)]["y"].sum()
    start_dt = date(2021, 1, 1)
    end_dt = datetime(2023, 1, 1)
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_output = get_sales_n_days_prior(df_comb, shift_days, "y", hourly=True)
    new_col = f"sales_{shift_days}_days_prior_y"
    assert new_col in df_output.columns
    future_dt = input_datetime + timedelta(days=shift_days)
    sales_check = df_output[(df_output["ds"] == future_dt)][new_col].sum()
    assert sales_check == sales_exp


@pytest.mark.parametrize(
    ("input_datetime", "shift_days"),
    (
            (datetime(2022, 1, 1), 7),
            (datetime(2022, 2, 1), 14),
            (datetime(2022, 6, 6), 28),
            (datetime(2022, 10, 7), 35),
    ),
)
def test_get_sales_n_days_prior(
        df_med_sales: pd.DataFrame, input_datetime: datetime, shift_days: int
) -> None:
    df_med_daily = aggregate_daily_sales(df_med_sales, 'ds', 'y')
    sales_exp = df_med_daily.loc[(df_med_daily["ds"] == input_datetime), "y"].values
    start_dt = date(2021, 1, 1)
    end_dt = datetime(2023, 1, 1)
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb_daily = aggregate_daily_sales(df_comb, 'ds', 'y')
    df_output = get_sales_n_days_prior(df_comb_daily, shift_days, "y")
    new_col = f"sales_{shift_days}_days_prior_y"
    assert new_col in df_output.columns
    future_dt = input_datetime + timedelta(days=shift_days)
    sales_check = df_output[(df_output["ds"] == future_dt)][new_col].sum()
    assert sales_check == sales_exp


def test_create_operation_flag_column(df_med_sales: pd.DataFrame) -> None:
    df_output = create_operation_flag_column(df_med_sales)
    assert "operational_flag" in df_output.columns
    assert df_output["operational_flag"].dtype == np.dtype("int64")


@pytest.mark.parametrize(
    ("start_dt", "dow", "dom", "month_id", "hour"),
    (
            (datetime(2022, 1, 1, 18), 5, 1, 1, 18),
            (datetime(2023, 2, 14, 19), 1, 14, 2, 19),
            (datetime(2022, 12, 25, 11), 6, 25, 12, 11),
    ),
)
def test_datetime_int_columns(
        df_med_sales: pd.DataFrame, start_dt, dow, dom, month_id, hour
) -> None:
    end_dt = start_dt
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb = create_dow_column(df_comb)
    assert "dow" in df_comb.columns
    assert df_comb["dow"][0] == dow
    df_comb = create_dom_column(df_comb)
    assert "dom" in df_comb.columns
    assert df_comb["dom"][0] == dom
    df_comb = create_month_id_column(df_comb)
    assert df_comb["month_id"][0] == month_id
    assert "month_id" in df_comb.columns
    df_comb = create_hour_column(df_comb)
    assert "hour" in df_comb.columns
    assert df_comb[df_comb["hour"] == hour]["hour"].values[0] == hour


@pytest.mark.parametrize(
    ("start_dt", "dow", "dow_cos", "dow_sin"),
    (
            (datetime(2022, 1, 3), 0, 1.000, 0.000),
            (datetime(2022, 1, 4), 1, 0.623, 0.782),
            (datetime(2022, 1, 5), 2, -0.223, 0.975),
            (datetime(2022, 1, 6), 3, -0.901, 0.434),
            (datetime(2022, 1, 7), 4, -0.901, -0.434),
            (datetime(2022, 1, 8), 5, -0.223, -0.975),
            (datetime(2022, 1, 9), 6, 0.623, -0.782),
    ),
)
def test_dow_cos_sin(
        df_med_sales: pd.DataFrame, start_dt, dow, dow_cos, dow_sin
) -> None:
    end_dt = start_dt
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb = create_dow_column(df_comb)
    assert df_comb["dow"][0] == dow
    assert df_comb["dow_cos"][0] == dow_cos
    assert df_comb["dow_sin"][0] == dow_sin


@pytest.mark.parametrize(
    ("start_dt", "month_id", "month_id_cos", "month_id_sin"),
    (
            (datetime(2022, 1, 2, 1), 1, 0.866, 0.5),
            (datetime(2022, 2, 2, 2), 2, 0.5, 0.866),
            (datetime(2022, 3, 2, 3), 3, 0.0, 1.0),
            (datetime(2022, 4, 2, 4), 4, -0.5, 0.866),
            (datetime(2022, 5, 2, 5), 5, -0.866, 0.5),
            (datetime(2022, 6, 2, 6), 6, -1.0, 0.0),
            (datetime(2022, 7, 2, 7), 7, -0.866, -0.5),
            (datetime(2022, 8, 2, 8), 8, -0.5, -0.866),
            (datetime(2022, 9, 2, 9), 9, -0.0, -1.0),
            (datetime(2022, 10, 2, 10), 10, 0.5, -0.866),
            (datetime(2022, 11, 2, 11), 11, 0.866, -0.5),
            (datetime(2022, 12, 2, 12), 12, 1.0, -0.0),
    ),
)
def test_month_id_cos_sin(
        df_med_sales: pd.DataFrame, start_dt, month_id, month_id_cos, month_id_sin
) -> None:
    end_dt = start_dt
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb = create_month_id_column(df_comb)
    assert df_comb["month_id"][0] == month_id
    assert df_comb["month_id_cos"][0] == month_id_cos
    assert df_comb["month_id_sin"][0] == month_id_sin


@pytest.mark.parametrize(
    ("start_dt", "hour", "hour_cos", "hour_sin"),
    (
            (datetime(2022, 1, 2, 0), 0, 1.0, 0.0),
            (datetime(2022, 1, 2, 1), 1, 0.966, 0.259),
            (datetime(2022, 1, 2, 2), 2, 0.866, 0.5),
            (datetime(2022, 1, 2, 3), 3, 0.707, 0.707),
            (datetime(2022, 1, 2, 4), 4, 0.5, 0.866),
            (datetime(2022, 1, 2, 5), 5, 0.259, 0.966),
            (datetime(2022, 1, 2, 6), 6, 0.0, 1.0),
            (datetime(2022, 1, 2, 7), 7, -0.259, 0.966),
            (datetime(2022, 1, 2, 8), 8, -0.5, 0.866),
            (datetime(2022, 1, 2, 9), 9, -0.707, 0.707),
            (datetime(2022, 1, 2, 10), 10, -0.866, 0.5),
            (datetime(2022, 1, 2, 11), 11, -0.966, 0.259),
            (datetime(2022, 1, 2, 12), 12, -1.0, 0.0),
    ),
)
def test_hour_cos_sin(
        df_med_sales: pd.DataFrame, start_dt, hour, hour_cos, hour_sin
) -> None:
    end_dt = start_dt
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb = create_hour_column(df_comb)
    assert df_comb["hour"][hour] == hour
    assert df_comb["hour_cos"][hour] == hour_cos
    assert df_comb["hour_sin"][hour] == hour_sin


@pytest.mark.parametrize(
    ("start_dt", "week_id"),
    (
            (datetime(2023, 1, 1), 52),
            (datetime(2023, 1, 8), 1),
    ),
)
def test_create_week_id_columns(
        df_med_sales: pd.DataFrame, start_dt, week_id
) -> None:
    end_dt = start_dt
    df_comb = create_full_sales_history(start_dt, end_dt, df_med_sales)
    df_comb = create_week_id_column(df_comb)
    assert df_comb["week_id"][0] == week_id


def test_x_day_forecast(df_med_sales: pd.DataFrame) -> None:
    d1 = datetime(2021, 1, 1)
    d2 = datetime(2021, 2, 12)
    offsets = [7, 14, 21, 28, 35, 42]
    d_offsets = [d1 + timedelta(days=d) for d in offsets]
    df_comb = create_full_sales_history(d1, d2, df_med_sales)
    sales_col = "y"
    df_daily = aggregate_daily_sales(df_comb, 'ds', [sales_col])
    for o in offsets:
        df_daily = get_sales_n_days_prior(df_daily, o, sales_col)

    df_comb = df_daily
    df_comb = n_week_forecast(df_comb, sales_col, 1)
    df_comb = n_week_forecast(df_comb, sales_col, 2)

    # 7 day manual
    _7_day_for_man = df_comb[df_comb["ds"].isin(d_offsets[1:-1])][sales_col].mean()
    logger.info(f"7 day forecast manual: {_7_day_for_man}")
    _14_day_input = df_comb[df_comb["ds"].isin(d_offsets[:4])][sales_col].mean()
    logger.info(f"14 day forecast input: {_14_day_input}")
    _14_day_sum = df_comb[df_comb["ds"].isin(d_offsets[1:-2])][sales_col].sum()
    logger.info(f"14 day sum 1-3: {_14_day_sum}")
    _14_day_for_man = (_14_day_input + _14_day_sum) / 4
    logger.info(f"14 day forecast manual: {_14_day_for_man}")

    _7_day_forecast_auto = df_comb[df_comb["ds"] == d2][f"forecast_7_days_{sales_col}"].values[0]
    logger.info(f"7 day forecast auto: {_7_day_forecast_auto}")
    _14_day_forecast_auto = df_comb[df_comb["ds"] == d2][f"forecast_14_days_{sales_col}"].values[0]
    logger.info(f"14 day forecast auto: {_14_day_forecast_auto}")

    assert abs(_7_day_forecast_auto - _7_day_for_man) < 0.1
    assert abs(_14_day_forecast_auto - _14_day_for_man) < 0.1


def test_x_day_forecast_with_zero_values() -> None:
    df = pd.DataFrame({f'sales_{i * 7}_days_prior_y': [1, 2, 3] for i in range(1, 9)})
    df.at[0, 'sales_7_days_prior_y'] = 0
    df.at[1, 'sales_7_days_prior_y'] = 0
    df.at[0, 'sales_14_days_prior_y'] = 0
    df.at[0, 'sales_21_days_prior_y'] = 0
    df.at[0, 'sales_28_days_prior_y'] = 0

    # Test five week forecast
    df.at[2, 'sales_7_days_prior_y'] = np.nan
    df.at[2, 'sales_14_days_prior_y'] = np.nan
    df.at[2, 'sales_21_days_prior_y'] = np.nan
    df.at[2, 'sales_28_days_prior_y'] = np.nan

    df_f = n_week_forecast(df, 'y', 1)
    df_f = n_week_forecast(df_f, 'y', 2)
    df_f = n_week_forecast(df_f, 'y', 4)
    df_f = n_week_forecast(df_f, 'y', 5)

    assert df_f.at[0, "forecast_7_days_y"] == 0
    assert df_f.at[1, "forecast_7_days_y"] == 2
    assert df_f.at[0, "forecast_14_days_y"] == 1
    assert df_f.at[0, "forecast_28_days_y"] == 1
    assert df_f.at[2, "forecast_35_days_y"] == 3


def test_n_day_forecast() -> None:
    df = pd.DataFrame({f'sales_{i * 7}_days_prior_y': [1, 2, 3] for i in range(1, 9)})
    df_exp = df.copy()
    df_exp['forecast_35_days_y'] = [1.0, 2.0, 3.0]
    df_act = n_week_forecast(df, 'y', 5)
    assert_frame_equal(df_act, df_exp)


def test_operational_flag(df_med_sales: pd.DataFrame) -> None:
    d1 = datetime(2020, 12, 31)
    d2 = datetime(2021, 1, 1)
    df = create_full_sales_history(d1, d2, df_med_sales)
    assert df["operational_flag"].sum() == 48  # 2 * 24hr = 48hr


def test_fill_const_columns() -> None:
    df = pd.DataFrame(
        {
            "a": ["a", "a", "a", np.nan],
            "b": [np.nan, np.nan, np.nan, np.nan],
            "c": ["c", np.nan, "c", "c"],
            "d": [np.nan, "a", "a", "a"],
        }
    )
    const_cols = ["a", "b", "c", "d"]
    df = fill_const_columns(df, const_cols)
    assert df.isna().sum().sum() == 0
    assert df.loc[3, "a"] == "a"
    assert df.loc[2, "b"] == ""
    assert df.loc[1, "c"] == "c"
    assert df.loc[0, "d"] == "a"


def test_drop_cos_sin_cols() -> None:
    df = pd.DataFrame(
        {
            "dow": ["a", "a", "a", np.nan],
            "dow_sin": [2, 2, np.nan, 2],
            "dow_cos": [3, np.nan, 3, 3],
        }
    )
    df = drop_cos_sin_cols(df)
    assert df.columns == ['dow']


def test_create_error_column() -> None:
    df = pd.DataFrame(
        {
            "y_adj": [1, 1, 1, 0, 0, 10],
            "forecast_7_days": [1, 0, 2, 0, 1, 1],
        }
    )
    df_output = create_error_column(df_input=df, sales_col="y_adj", forecast_col="forecast_7_days")
    assert 'error_7_days' in df_output.columns
    assert df_output.loc[0, 'error_7_days'] == 0.0
    assert np.isnan(df_output.loc[1, 'error_7_days'])
    assert df_output.loc[2, 'error_7_days'] == -1.0
    assert df_output.loc[3, 'error_7_days'] == 0.0
    assert df_output.loc[4, 'error_7_days'] == -1.0e6
    assert df_output.loc[5, 'error_7_days'] == 0.9


def test_create_year_column() -> None:
    df_hol = pd.DataFrame(
        {
            "ds": [datetime(2021, 12, 25, 10), datetime(2021, 12, 28, 23)],
            "cc": ['GB', 'MT'],
            "hol": ['xmas_uk', 'holiday']
        },
    )
    df_hol['ds'] = pd.to_datetime(df_hol['ds'])
    df_output = create_year_column(df_hol)

    assert df_output.loc[0, 'year'] == 2021


@pytest.mark.parametrize(
    ("input", "expected"),
    (
            ({"start_dt": datetime(2023, 11, 1), "n": 1}, {"rows": 1, "end_dt": datetime(2023, 11, 1)}),
            ({"start_dt": datetime(2023, 11, 1), "n": 7}, {"rows": 7, "end_dt": datetime(2023, 11, 7)}),
    )
)
def test_create_n_days_empty_sales(input, expected) -> None:
    expected_cols = ['unique_id', 'y', 'y_adj', 'operational_flag', 'is_forecast']
    print(input)
    df = create_n_days_empty_sales("dummy_id", input['start_dt'], input['n'])
    assert df.shape[0] == expected['rows']
    max_dt_hr = df['ds'].max()
    max_dt = datetime.strptime(max_dt_hr.strftime("%Y-%m-%d"), "%Y-%m-%d")
    assert max_dt == expected['end_dt']
    assert df.isna().sum().sum() == 0
    assert df.loc[0, 'forecast_horizon'] == 1
    assert df.iloc[-1]['forecast_horizon'] == expected['rows']


def test_append_n_days_of_empty_sales_on_historical_sales() -> None:
    df_sales = pd.DataFrame(
        {
            'unique_id': ['test', 'test', 'test'],
            'y': [10, 11, 12],
            'y_adj': [10, 11, 12],
            'y_adj_capped': [10, 11, 12],
            'operational_flag': [1, 1, 1],
            'ds': [datetime(2023, 10, 29), datetime(2023, 10, 30), datetime(2023, 10, 31)]
        }
    )
    s_date = datetime(2023, 11, 1)
    n = 1
    df_emtpy = create_n_days_empty_sales('test', s_date, n)
    df_output = append_n_days_of_empty_sales_on_historical_sales(df_sales, df_emtpy)

    assert df_output.shape[0] == df_sales.shape[0] + n
    max_dt_hr = df_output['ds'].max()
    max_dt = datetime.strptime(max_dt_hr.strftime("%Y-%m-%d"), "%Y-%m-%d")
    assert max_dt == s_date + timedelta(n - 1)
    assert df_output.isna().sum().sum() == 3  # Expect 3 nans in forecast_horizon_column
    assert df_output.shape[1] == 8


def test_create_perc_err_column() -> None:
    df_sales = pd.DataFrame(
        {
            "y": [10.0, 16.0, 20.0, 0.0, 10.0, np.nan, 1.0],
            "y_hat": [8.0, 20.0, 20.0, 10.0, 0.0, 0.0, 2],
        }
    )

    df_output = create_perc_err_column(df_sales, "y", "y_hat", "error")

    assert list(df_output["error"].values) == [0.2, -0.25, 0.0, 0.0, 1.0, 0.0, -1.0]


def test_aggregate_daily_sales(dir_testing_data) -> None:
    df = read_sales_from_disk(dir_testing_data / "dummy_sales.csv")
    df = zero_small_and_negatives_sales(df)
    s_date = datetime(2020, 12, 31)
    e_date = datetime(2021, 1, 2)
    df = create_full_sales_history(s_date, e_date, df)
    df_output = aggregate_daily_sales(df, 'ds', ['y_adj'])

    df_exp = pd.DataFrame({"unique_id": ["abc", "abc", "abc"],
                           'ds': ["2020-12-31", "2021-01-01", "2021-01-02"],
                           "y_adj": [20.0, 40.0, 10.0],
                           "operational_flag": [1.0, 1.0, 1.0]
                           })
    df_exp['ds'] = pd.to_datetime(df_exp['ds'])

    assert_frame_equal(df_output[['y_adj']], df_exp[['y_adj']])
    assert_frame_equal(df_output, df_exp)


def test_create_date_only_column(dir_testing_data) -> None:
    df = read_sales_from_disk(dir_testing_data / "dummy_sales.csv")
    df_output = create_date_only_column(df)
    df_output.drop_duplicates(subset=['ds_dt'], inplace=True)
    df_output.reset_index(inplace=True)

    df_exp = pd.DataFrame({'ds_dt': ['2020-12-31', "2021-01-01", "2021-01-02"]})
    df_exp['ds_dt'] = pd.to_datetime(df_exp['ds_dt'])

    assert_frame_equal(df_output[['ds_dt']], df_exp)


def test_count_positive_sales_days_in_last_n_days(dir_testing_data) -> None:
    df = pd.read_csv(dir_testing_data / "dummy_sales_2.csv")
    df['ds'] = pd.to_datetime(df['ds'])

    df_output = count_positive_sales_days_in_last_n_days(df, 7, "y")

    df_exp = pd.DataFrame({
        'positive_sale_days_in_last_7_days': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 6.0, 5.0, 4.0, ]
    })

    assert_frame_equal(df_output[['positive_sale_days_in_last_7_days']], df_exp)


def test_create_sales_cap_column() -> None:
    y = [1, 2, 3] * 10 + [6, 7, 8, 9, 10]
    df = pd.DataFrame({'y': y})
    df_output = create_sales_cap_column(df_sales=df, sales_col='y', std_above_median=3)
    y_exp = [1, 2, 3] * 10 + [6, 7, 8, 9, 9]
    df_exp = pd.DataFrame({'y': y, 'y_capped': y_exp})
    assert_frame_equal(df_output, df_exp)


@pytest.mark.parametrize(
    ("input", "expected"),
    (
            ("smear_test_incomplete_year.csv", "smear_test_incomplete_year_exp.csv"),
            ("smear_test_single_month.csv", "smear_test_single_month_exp.csv"),
    ),
)
def test_calculate_hourly_avg_contribution(dir_testing_data, input, expected) -> None:
    f_path = dir_testing_data / input
    f_path_exp = dir_testing_data / expected
    df1 = read_sales_from_disk(f_path)
    df_exp = pd.read_csv(f_path_exp)
    df_act = calculate_hourly_avg_contribution(
        df_hourly_sales=df1,
        date_col='ds',
        e_d=datetime(2023, 12, 5),
    )
    assert_frame_equal(df_act, df_exp, check_dtype=False)


def test_left_anti_join() -> None:
    df_l = pd.DataFrame({'dow_id': [1, 2, 3, 4, 5], 'month_id': [1, 2, 3, 4, 5],
                         "hour_id": [1, 1, 1, 1, 1], "avg_hour_contribution": [0.5, 0.5, 0.5, 0.5, 0.5, ]})
    df_r = pd.DataFrame({'dow_id': [1, 3], 'month_id': [1, 3],
                         "hour_id": [1, 1], "avg_hour_contribution": [0.6, 0.7]})

    df_act = left_anti_join(df_l, df_r, ['dow_id', 'month_id', 'hour_id'])

    df_exp = pd.DataFrame({'dow_id': [2, 4, 5], 'month_id': [2, 4, 5],
                           "hour_id": [1, 1, 1], "avg_hour_contribution": [0.5, 0.5, 0.5]}, index=[0, 1, 2])

    assert_frame_equal(df_act, df_exp)


@pytest.fixture
def input_for_hol_sf() -> pd.DataFrame:
    test_df = pd.DataFrame(
        {
            "sf_7_days_y_adj_capped_branch_median": [30, 1.0, 1.0, 1.0],
            "sf_7_days_y_adj_capped_brand_median": [0.1, 0.01, 1.0, 1.0],
            "sf_7_days_y_adj_capped_city_median": [1.0, 2.0, 0.1, 1.0],
            "sf_7_days_y_adj_capped_country_median": [1.0, 1.0, 3.0, 0.1],
        }
    )
    return test_df


@pytest.fixture
def exp_hol_sf_df(input_for_hol_sf) -> pd.DataFrame:
    test_df_exp = input_for_hol_sf
    test_df_exp["sf_final_7_days_y_adj_capped"] = [5, 0.1, 0.1, 0.1]
    test_df_exp["sf_final_7_days_y_adj_capped_label"] = [
        "sf_7_days_y_adj_capped_branch_median",
        "sf_7_days_y_adj_capped_brand_median",
        "sf_7_days_y_adj_capped_city_median",
        "sf_7_days_y_adj_capped_country_median",
    ]
    return test_df_exp


@pytest.fixture
def baseline_sf_df() -> pd.DataFrame:
    test_df_baseline = pd.DataFrame(
        {
            "sf_7_days_y_adj_capped_branch_median": [1, 1, 1, 1],
            "sf_7_days_y_adj_capped_brand_median": [1, 1, 1, 1],
            "sf_7_days_y_adj_capped_city_median": [1, 1, 1, 1],
            "sf_7_days_y_adj_capped_country_median": [1, 1, 1, 1],
        }
    )
    return test_df_baseline


@pytest.fixture
def exp_baseline_sf_df(baseline_sf_df) -> pd.DataFrame:
    test_df_baseline_exp = baseline_sf_df
    test_df_baseline_exp["sf_final_7_days_y_adj_capped"] = [1, 1, 1, 1]
    test_df_baseline_exp["sf_final_7_days_y_adj_capped_label"] = [
        "sf_7_days_y_adj_capped_none",
        "sf_7_days_y_adj_capped_none",
        "sf_7_days_y_adj_capped_none",
        "sf_7_days_y_adj_capped_none",
    ]
    return test_df_baseline_exp


@pytest.mark.parametrize(
    ("input_value", "expected", "min_sf", "max_sf"),
    (
            pytest.param(
                "input_for_hol_sf", "exp_hol_sf_df", 0.1, 5
            ),
            pytest.param("baseline_sf_df", "exp_baseline_sf_df", None, None),
    ),
)
def test_create_optimal_hol_sf_column(
        input_value: pd.DataFrame,
        expected: pd.DataFrame,
        min_sf: float | None,
        max_sf: float | None,
        request,
) -> None:
    input_value = request.getfixturevalue(input_value)
    actual = create_optimal_hol_sf_column(
        df_input=input_value,
        forecast_horizon_days=7,
        sales_type="y_adj_capped",
        min_sf=min_sf,
        max_sf=max_sf,
    )
    actual = create_label_optimal_hol_sf_column(
        df_input=actual, forecast_horizon_days=7, sales_type="y_adj_capped"
    )
    expected = request.getfixturevalue(expected)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("sale", "forecast", "zero_sales_zero_error_flag", "expected"),
    (
            pytest.param(20, 19, True, 0.05, id="under forecast"),
            pytest.param(20, 21, True, -0.05, id="over forecast"),
            pytest.param(20, 20, True, 0.00, id="exact forecast"),
            pytest.param(20, 0, True, np.nan, id="actual sales but zero forecast"),
            pytest.param(0, 19, True, 0.00, id="zero sales, zero error"),
            pytest.param(0, 19, False, -1e6, id="zero sales, large neg error -> -1e6"),
            pytest.param(0, 0, True, 0.0, id="zero sales, zero forecast, exact forecast"),
    ),
)
def test_create_error_column(
        sale, forecast, zero_sales_zero_error_flag, expected
) -> None:
    df = pd.DataFrame(
        {
            "sales_adj_capped": [sale],
            "forecast_7_days_sales_adj_capped": [forecast],
        }
    )
    df_output = create_error_column(
        df_input=df,
        sales_col="sales_adj_capped",
        forecast_col="forecast_7_days_sales_adj_capped",
        zero_sales_zero_error=zero_sales_zero_error_flag,
    )
    if sale > 0 and forecast == 0:
        assert np.isnan(df_output.loc[0, "error_7_days_sales_adj_capped"])
    else:
        assert df_output.loc[0, "error_7_days_sales_adj_capped"] == expected
