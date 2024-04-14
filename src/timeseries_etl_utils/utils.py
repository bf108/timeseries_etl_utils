from datetime import datetime, date, timedelta
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def convert_daily_hols_to_hours(df_hols: pd.DataFrame) -> pd.DataFrame:
    rows_prior = df_hols.shape[0]
    assert "date" in df_hols.columns
    min_dt = df_hols["date"].min()
    max_dt = df_hols["date"].max()
    df_hours = create_hourly_step_calendar(min_dt, max_dt)
    df_hours["date"] = pd.to_datetime(df_hours["ds"].dt.strftime("%Y-%m-%d"))
    df_output = df_hols.merge(df_hours, how="left", on="date")
    assert (
            df_output.shape[0] == rows_prior * 24
    ), f"{df_output.shape[0]} {rows_prior * 24}"
    return df_output


def get_zero_hrs_of_date(dt: date) -> date:
    return datetime.strptime(dt.strftime("%Y-%m-%d"), "%Y-%m-%d")


def get_last_hrs_of_date(dt: date) -> date:
    return datetime.strptime(dt.strftime("%Y-%m-%d"), "%Y-%m-%d") + timedelta(
        hours=23
    )


def create_hourly_step_calendar(start_dt: date, end_dt: date) -> pd.DataFrame:
    s_dt = get_zero_hrs_of_date(start_dt)
    e_dt = get_last_hrs_of_date(end_dt)
    df_calendar = pd.DataFrame(pd.date_range(s_dt, e_dt, freq="h")).rename(
        columns={0: "ds"}
    )
    return df_calendar


def create_daily_step_calendar(start_dt: date, end_dt: date) -> pd.DataFrame:
    s_dt = get_zero_hrs_of_date(start_dt)
    e_dt = get_last_hrs_of_date(end_dt)
    df_calendar = pd.DataFrame(pd.date_range(s_dt, e_dt, freq="D")).rename(
        columns={0: "ds"}
    )
    return df_calendar


def read_sales_from_disk(f_path: Path) -> pd.DataFrame:
    sales_df = pd.read_csv(f_path, parse_dates=['ds'])
    df_output = create_operation_flag_column(sales_df)
    return df_output


def read_hol_from_disk(f_path: Path) -> pd.DataFrame:
    return pd.read_csv(f_path, parse_dates=["date"])


def read_opening_hours_from_disk(f_path: Path) -> pd.DataFrame:
    df = pd.read_csv(f_path, usecols=["unique_id", "dow", "hour", "is_open_flag"])
    df["is_open_flag"] = df["is_open_flag"].astype(bool)
    return df


def read_venue_geo_from_disk(f_path: Path) -> pd.DataFrame:
    df_geo = pd.read_csv(f_path)
    return df_geo


def create_full_sales_history(
        start_dt: date, end_dt: date, df_sales: pd.DataFrame
) -> pd.DataFrame:
    u_id = df_sales["unique_id"].unique()[0]
    df_cal = create_hourly_step_calendar(start_dt, end_dt)
    df_comb = df_cal.merge(df_sales, how="left", on="ds").fillna(
        value={"y": 0, "unique_id": u_id, "y_adj": 0}
    )
    df_comb["operational_flag"] = 1.0
    cal_rows = df_cal.shape[0]
    comb_rows = df_comb.shape[0]
    assert (
            cal_rows == comb_rows
    ), f"Join issue between hrly calendar and sales.\nhrly cal rows: {cal_rows},\nrows after left join: {comb_rows}"
    return df_comb


def create_n_days_empty_sales(
        unique_id: str, start_dt: datetime, n: int
) -> pd.DataFrame:
    assert n >= 1, "n must be an int greater than or equal to one"
    end_dt = start_dt + timedelta(days=n - 1)
    df_output = create_daily_step_calendar(start_dt, end_dt)
    df_output.loc[:, 'unique_id'] = unique_id
    df_output.loc[:, 'y'] = 0
    df_output.loc[:, 'y_adj'] = 0
    df_output.loc[:, 'y_adj_capped'] = 0
    df_output.loc[:, 'operational_flag'] = 1
    df_output.loc[:, 'is_forecast'] = True
    df_output = df_output.reset_index().rename(columns={'index': 'forecast_horizon'})
    df_output['forecast_horizon'] += 1
    return df_output


def append_n_days_of_empty_sales_on_historical_sales(
        df_sales: pd.DataFrame, df_emtpy_sales: pd.DataFrame
) -> pd.DataFrame:
    df_output = pd.concat([df_sales, df_emtpy_sales])
    df_output['is_forecast'] = df_output['is_forecast'].astype(bool).fillna(False)
    df_output.reset_index(drop=True, inplace=True)
    df_output.sort_values(by='ds', ascending=True)
    return df_output


def zero_small_and_negatives_sales(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["y_adj"] = df_output["y"]
    df_output.loc[df_output["y"] < 1, "y_adj"] = 0
    return df_output


def get_sales_n_days_prior(
        df_input: pd.DataFrame, n: int, col_to_shift: str, hourly: bool = False,
) -> pd.DataFrame:
    _n = n
    freq = 'D'
    if hourly:
        _n = 24 * n
        freq = 'H'
    assert col_to_shift in df_input.columns, f"{col_to_shift} not in {df_input.columns}"
    assert "ds" in df_input.columns, f"ds not in {df_input.columns}"
    df_output = df_input.copy()
    df_output.set_index("ds", inplace=True)
    new_col = f"sales_{n}_days_prior_{col_to_shift}"
    df_output[new_col] = df_output[col_to_shift].shift(periods=_n, freq=freq)
    df_output.reset_index(inplace=True)
    return df_output


def create_operation_flag_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["operational_flag"] = 1
    return df_output


def create_dom_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["dom"] = df_output["ds"].dt.day
    df_output["dom_cos"] = np.round(np.cos(df_output["dom"] * (2 * np.pi / 31)), 3)
    df_output["dom_sin"] = np.round(np.sin(df_output["dom"] * (2 * np.pi / 31)), 3)
    return df_output


def create_dow_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["dow"] = df_output["ds"].dt.dayofweek
    df_output["dow_cos"] = np.round(np.cos(df_output["dow"] * (2 * np.pi / 7)), 3)
    df_output["dow_sin"] = np.round(np.sin(df_output["dow"] * (2 * np.pi / 7)), 3)
    return df_output


def create_month_id_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["month_id"] = df_output["ds"].dt.month
    df_output["month_id_cos"] = np.round(
        np.cos(df_output["month_id"] * (2 * np.pi / 12)), 3
    )
    df_output["month_id_sin"] = np.round(
        np.sin(df_output["month_id"] * (2 * np.pi / 12)), 3
    )
    return df_output


def create_hour_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["hour"] = df_output["ds"].dt.hour
    df_output["hour_cos"] = np.round(np.cos(df_output["hour"] * (2 * np.pi / 24)), 3)
    df_output["hour_sin"] = np.round(np.sin(df_output["hour"] * (2 * np.pi / 24)), 3)
    return df_output


def create_week_id_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["week_id"] = df_output["ds"].dt.isocalendar().week
    df_output["week_id_cos"] = np.round(
        np.cos(df_output["week_id"] * (2 * np.pi / 52)), 3
    )
    df_output["week_id_sin"] = np.round(
        np.sin(df_output["week_id"] * (2 * np.pi / 52)), 3
    )
    return df_output


def create_year_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output["year"] = df_output["ds"].dt.year
    return df_output


def create_date_only_column(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_output = df_sales.copy()
    df_output['ds_dt'] = pd.to_datetime(df_output['ds'].dt.date)
    return df_output


def create_date_feat_cols(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output = create_dow_column(df_output)
    df_output = create_dom_column(df_output)
    df_output = create_week_id_column(df_output)
    df_output = create_month_id_column(df_output)
    return df_output


def count_positive_sales_days_in_last_n_days(df_sales: pd.DataFrame, n: int, sales_col: str) -> pd.DataFrame:
    assert isinstance(n, int)
    df_output = df_sales.copy()
    df_output["positive_sales_flag"] = 0
    df_output.loc[(df_output["operational_flag"] == 1) & (df_output[sales_col] > 0), "positive_sales_flag"] = 1
    df_output[f"positive_sale_days_in_last_{n}_days"] = (
        df_output["positive_sales_flag"].rolling(min_periods=1, window=n).sum()
    )
    return df_output


def n_week_forecast(df_input: pd.DataFrame, sales_col: str, n: int) -> pd.DataFrame:
    df_output = df_input.copy()
    cols = [f"sales_{(i * 7)}_days_prior_{sales_col}" for i in range(n, n + 4)]
    for c in cols:
        assert c in df_output.columns
    col_adj = cols
    if n > 1:
        latent_forecast = [i for i in range(n - 1, 0, -1)]
        for latent_input in latent_forecast:
            df_output = average_sales_days(df_output, col_adj, f"input_{latent_input}_week_forecast")
            col_adj = [f"input_{latent_input}_week_forecast"] + col_adj[:-1]
    df_output = average_sales_days(df_output, col_adj, f"forecast_{n * 7}_days_{sales_col}")
    if n > 1:
        df_output.drop(columns=col_adj[:len(latent_forecast)], inplace=True)
    return df_output


def average_sales_days(df: pd.DataFrame, list_days: list[str], col_name: str) -> pd.DataFrame:
    df_output = df.copy()
    df_output[col_name] = df_output[list_days].replace(0, np.nan).mean(axis=1).fillna(0)
    return df_output


def create_forecast_from_hol_sf(*, df_input: pd.DataFrame, forecast_suffixes: list[str]):
    df_output = df_input.copy()
    for suffix in forecast_suffixes:
        for day in [7, 14, 28, 35]:
            for sf_suffix in ["country_median", "city_median", "brand_median", "branch_median"]:
                original_forecast_col = f"forecast_{day}_days_{suffix}"
                new_col = f"{original_forecast_col}_sf_{sf_suffix}"
                sf_col = f"sf_{day}_days_{suffix}_{sf_suffix}"
                df_output[new_col] = df_output[original_forecast_col] * df_output[sf_col]
    return df_output


def generate_features(
        fp_sales: Path,
        df_country_sf: pd.DataFrame,
        df_city_sf: pd.DataFrame,
        df_brandname_sf: pd.DataFrame,
        df_branchname_sf: pd.DataFrame,
        fp_opening_hours: Path,
        fp_geo: Path,
        fp_hols: Path,
        fp_uuid_exc: Path,
        end_dt: datetime,
        min_hol_sf: float,
        max_hol_sf: float,
) -> pd.DataFrame:
    df_sales = read_sales_from_disk(fp_sales)
    df_opening_hours = read_opening_hours_from_disk(fp_opening_hours)
    df_geo = read_venue_geo_from_disk(fp_geo)
    df_hols = read_hol_from_disk(fp_hols)
    df_hols_hrs = convert_daily_hols_to_hours(df_hols)
    df_exc = pd.read_csv(fp_uuid_exc)

    # Opted for an inner join to ensure that we only build features for venues in nory.branches.json
    df_sales = df_sales[~df_sales['unique_id'].isin(list(df_exc['unique_id'].values))]

    dfs = []
    tot = df_sales["unique_id"].nunique()
    sales_raw = "y"
    sales_adj = "y_adj"
    sales_adj_capped = sales_adj + "_capped"
    for i, u_id in enumerate(df_sales["unique_id"].unique()):
        if u_id in df_geo['unique_id'].values and i % 25 == 0:
            logger.info(f"Working on {i + 1} out of {tot}")
        df_tmp = df_sales[df_sales["unique_id"] == u_id].copy()
        df = zero_small_and_negatives_sales(df_tmp)
        # Make sure full sales
        start_dt = df_tmp['ds'].min()
        df = create_full_sales_history(start_dt, end_dt, df)
        df_daily = aggregate_daily_sales(df, 'ds', [sales_raw, sales_adj])
        df_daily_capped = create_sales_cap_column(df_sales=df_daily, sales_col=sales_adj, std_above_median=3.0)
        df = df_daily_capped.merge(df_geo, how="inner", on="unique_id")
        # Create 28 days of empty sales data
        start_dt_empty = end_dt + timedelta(days=1)
        df_empty = create_n_days_empty_sales(u_id, start_dt_empty, 35)
        df = append_n_days_of_empty_sales_on_historical_sales(df, df_empty)

        # Add in count of positive sales days in last 28
        df = count_positive_sales_days_in_last_n_days(df_sales=df, n=28, sales_col=sales_adj_capped)

        # Add in count of positive sales days in last 56
        df = count_positive_sales_days_in_last_n_days(df_sales=df, n=56, sales_col=sales_adj_capped)

        # Fill Down Numeric columns
        num_cols_fill = ["latitude", "longitude", "dist_to_weather_loc_km", "weather_location_id"]
        for col in num_cols_fill:
            df[col] = df[col].ffill()

        # Fill const string columns
        const_cols = ["brandname", "branchname", "cc", "city", "admin1",
                      "admin2", "weather_map_location", "timezone_location"]
        df = fill_const_columns(df, const_cols)

        # Get historic sales
        days_used_for_sma_cals = [7, 14, 21, 28, 35, 42, 49, 56]
        # 364 days
        same_dow_1_yr_prior = [52 * 7]
        leading_month_1_yr_prior = [364 - 7 * i for i in range(1, 5)]
        trailing_month_1_yr_prior = [364 + 7 * i for i in range(1, 5)]
        days = days_used_for_sma_cals + same_dow_1_yr_prior + leading_month_1_yr_prior + trailing_month_1_yr_prior
        for d in days:
            df = get_sales_n_days_prior(df, d, sales_raw, hourly=False)
            df = get_sales_n_days_prior(df, d, sales_adj, hourly=False)
            df = get_sales_n_days_prior(df, d, sales_adj_capped, hourly=False)

        # Create date feature columns
        df = create_year_column(df)
        df = create_date_feat_cols(df)

        # Join holiday SF
        df = df.merge(df_hols_hrs, how='left', on=['ds', 'cc'])
        df = df.merge(df_country_sf, how='left', on=['year', 'cc', 'holiday_name'])
        df = df.merge(df_city_sf, how='left', on=['year', 'cc', 'city', 'holiday_name'])
        df = df.merge(df_brandname_sf, how='left', on=['year', 'brandname', 'holiday_name'])
        df = df.merge(df_branchname_sf, how='left', on=['year', 'unique_id', 'holiday_name'])

        fill_values = {col: 1 for col in df.columns if col.startswith("sf_")}
        df.fillna(value=fill_values, inplace=True)

        for d in [7, 35]:
            df = create_optimal_hol_sf_column(df_input=df,
                                              forecast_horizon_days=d,
                                              sales_type=sales_adj_capped,
                                              min_sf=min_hol_sf,
                                              max_sf=max_hol_sf
                                              )
            df = create_label_optimal_hol_sf_column(df_input=df, forecast_horizon_days=d, sales_type=sales_adj_capped)

        # Calculate average trailing monthly sales per dow one year prior
        col_month_prior = "y_adj_capped_1m_trailing_1_yr_prior"
        cols = [f"sales_{day}_days_prior_{sales_adj_capped}" for day in trailing_month_1_yr_prior]
        df = average_sales_days(df, cols, col_month_prior)

        # Calculate average leading monthly sales per dow one year prior
        col_month_leading = "y_adj_capped_1m_leading_1_yr_prior"
        cols = [f"sales_{day}_days_prior_{sales_adj_capped}" for day in leading_month_1_yr_prior]
        df = average_sales_days(df, cols, col_month_leading)

        # Calculate dir of sales
        df['capped_sales_trend'] = ((df[col_month_leading] -
                                     df[col_month_prior]) / df[col_month_prior]).replace(np.inf, 1)

        # sma Forecasts
        for j in range(1, 6):
            for col in [sales_raw, sales_adj, sales_adj_capped]:
                df = n_week_forecast(df, col, j)

        df = create_forecast_from_hol_sf(df_input=df, forecast_suffixes=[sales_adj_capped])

        # Create forecast from optimal sf
        for d in [7, 35]:
            df[f'forecast_{d}_days_y_adj_capped_sf_optimal'] = (
                    df['forecast_7_days_y_adj_capped'] * df[f'sf_final_{d}_days_y_adj_capped']
            )

        dfs.append(df)

    df_final = pd.concat(dfs)
    logger.info(f"Total rows of df_final: {df_final.shape}")

    # Keep only data after operational
    df_final = df_final[df_final["operational_flag"] == 1].copy()
    logger.info(f"Total rows after only keeping operational rows: {df_final.shape}")

    # Join on opening hours
    # df_final = df_final.merge(
    #     df_opening_hours, how="left", on=["unique_id", "dow", "hour"]
    # )
    return df_final


def fill_const_columns(df_input: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    fill_values = {}
    for col in cols:
        assert (
                col in df_input.columns
        ), f"{col} not in {df_input.columns}, so can't fill"
        assert (
                df_input[col].nunique() <= 1
        ), f"{col} does not have unique value, has: {df_input[col].nunique()}"
        # Find first value which isn't np.nan
        if len(df_input[col].unique()) > 1:
            fill_val = [val for val in df_input[col].unique() if val is not np.nan][0]
        else:
            fill_val = df_input[col].unique()[0]
        # if column is entirely np.nan then fill with "", rather than np.nan
        if not isinstance(fill_val, str):
            fill_val = ""
        fill_values[col] = fill_val
    df_output = df_input.copy()
    df_output.fillna(value=fill_values, inplace=True)
    return df_output


def drop_cos_sin_cols(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    to_drop = [col for col in df_output.columns if (("_cos" in col) or ("_sin" in col))]
    df_output.drop(columns=to_drop, inplace=True)
    return df_output


def create_error_column(*, df_input: pd.DataFrame, sales_col: str,
                        forecast_col: str, zero_sales_zero_error: bool = False) -> pd.DataFrame:
    """This is an error based on the simple 4-week rolling avg and actual sales"""
    # Case when zero forecast, but sale > 0. Current method to ignore. Leave error as default 0
    df_output = df_input.copy()
    df_output_cols = df_output.columns
    assert sales_col in df_output_cols, f"{sales_col} missing from {df_output_cols}"
    assert (
            f"{forecast_col}" in df_output_cols
    ), f"{forecast_col} missing from {df_output_cols}"

    error_suffix = forecast_col.split("forecast_")[1]
    # Default error set to zero
    df_output[f"error_{error_suffix}"] = 0.0
    # Actual sales > 0, but forecast zero > np.nan
    df_output.loc[
        ((df_output[sales_col] > 0) & (df_output[forecast_col] == 0)), f"error_{error_suffix}"
    ] = np.nan
    if not zero_sales_zero_error:
        # Case when zero sales and forecast > 0. Set error to large neg number. This will force SF -> 0
        df_output.loc[
            ((df_output[sales_col] == 0) & (df_output[forecast_col] > 0)), f"error_{error_suffix}"
        ] = -1.0e6
    # Case when sales and forecast > 0. Normal error calculation.
    df_output.loc[
        ((df_output[sales_col] > 0) & (df_output[forecast_col] > 0)), f"error_{error_suffix}"
    ] = (df_output[sales_col] - df_output[forecast_col]) / df_output[sales_col]
    return df_output


def create_perc_err_column(
        df_input: pd.DataFrame, actual: str, forecast: str, error_col: str
) -> pd.DataFrame:
    assert actual in df_input.columns
    assert forecast in df_input.columns
    df_output = df_input.copy()
    df_output[error_col] = 0.0
    df_output.loc[(df_output[actual] > 0), error_col] = (
                                                                df_output[actual] - df_output[forecast]
                                                        ) / df_output[actual]
    return df_output


def aggregate_daily_sales(df_input: pd.DataFrame, date_col: str, sales_cols: list[str]) -> pd.DataFrame:
    agg_dict = {f"{sc}": pd.NamedAgg(column=sc, aggfunc="sum") for sc in sales_cols}
    agg_dict["operational_flag"] = pd.NamedAgg(column="operational_flag", aggfunc="max")
    df_output = df_input.groupby(by=['unique_id', df_input[date_col].dt.date]).agg(**agg_dict)
    df_output.reset_index(inplace=True)
    df_output[date_col] = pd.to_datetime(df_output[date_col])
    return df_output


def create_sales_cap_column(*, df_sales: pd.DataFrame, sales_col: str, std_above_median: float = 3.0) -> pd.DataFrame:
    df_output = df_sales.copy()
    _median = df_output[(df_output[sales_col] > 0)][sales_col].median()
    _std = df_output[(df_output[sales_col] > 0)][sales_col].std()
    _max = np.round((_median + std_above_median * _std), 0)
    df_output[f"{sales_col}_capped"] = df_output[sales_col]
    df_output.loc[(df_output[sales_col] > _max), f"{sales_col}_capped"] = _max
    return df_output


def generate_input_hol_sf(
        fp_sales: Path,
        fp_geo: Path,
        end_dt: datetime,
        uuid_ignore: list[str],
) -> pd.DataFrame:
    df_sales = read_sales_from_disk(fp_sales)
    df_sales_ = df_sales[~df_sales['unique_id'].isin(uuid_ignore)].copy()
    df_geo = read_venue_geo_from_disk(fp_geo)
    dfs = []
    tot = df_sales_["unique_id"].nunique()
    sales_adj = "y_adj"
    sales_adj_capped = sales_adj + "_capped"
    for i, u_id in enumerate(df_sales_["unique_id"].unique()):
        if u_id in df_geo['unique_id'].values:
            if i % 25 == 0:
                logger.info(f"Working on {i + 1} out of {tot}")
            df_tmp = df_sales_[df_sales_["unique_id"] == u_id].copy()
            df_ = zero_small_and_negatives_sales(df_tmp)
            start_dt = df_tmp['ds'].min()
            df_ = create_full_sales_history(start_dt, end_dt, df_)
            df_daily = aggregate_daily_sales(df_, 'ds', [sales_adj])
            df_daily_capped = create_sales_cap_column(df_sales=df_daily, sales_col=sales_adj, std_above_median=3.0)
            df = df_daily_capped.merge(df_geo, how="inner", on="unique_id")
            # Get historic sales
            days = [7, 14, 21, 28, 35, 42, 49, 56]
            for d in days:
                df = get_sales_n_days_prior(df, d, sales_adj_capped, hourly=False)
            # sma Forecasts
            for j in range(1, 6):
                df = n_week_forecast(df, sales_adj_capped, j)

            # Drop sales history after generating forecast
            df.drop(columns=[f"sales_{day}_days_prior_{sales_adj_capped}" for day in days], inplace=True)
            # Date columns
            df = create_dow_column(df)
            df = create_hour_column(df)
            # Drop cos/sin columns
            df = drop_cos_sin_cols(df)
            for day in [7, 14, 28, 35]:
                forecast_col = f"forecast_{day}_days_{sales_adj_capped}"
                df = create_error_column(df_input=df, sales_col=sales_adj_capped,
                                         forecast_col=forecast_col, zero_sales_zero_error=False)
            dfs.append(df)
        else:
            logger.info(f"Excluding {u_id}, as no meta data available")

    df_final = pd.concat(dfs)
    # Keep only data after operational
    df_final = df_final[df_final["operational_flag"] == 1].copy()
    df_final = create_year_column(df_final)
    # Select relevant columns for hol sf calc and useful columns for later assessment and debugging
    cols_of_interest = ["ds", "unique_id", "branchname", "brandname", "year", "city", "cc"]
    error_cols = [col for col in df_final if col.startswith("error_")]
    df_final = df_final.loc[:, cols_of_interest + error_cols]
    return df_final


def prepare_sales_for_hourly_smear_agg(*, df_hourly_sales: pd.DataFrame,
                                       s_d: datetime, e_d: datetime, date_col: str) -> pd.DataFrame:
    df_output = df_hourly_sales.copy()
    df_output = create_full_sales_history(s_d, e_d, df_output)
    df_output = df_output[df_output['operational_flag'] == 1].copy()
    df_output = zero_small_and_negatives_sales(df_output)
    df_output = create_basic_dt_cols(df_input=df_output, date_col=date_col)
    return df_output


def create_basic_dt_cols(*, df_input: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output['month_id'] = df_output[date_col].dt.month
    df_output['dow_id'] = df_output[date_col].dt.dayofweek
    df_output['hour_id'] = df_output[date_col].dt.hour
    df_output['dt_str'] = df_output[date_col].dt.date
    return df_output


def create_month_dow_hr_df():
    month_id_list = []
    dow_id_list = []
    hour_id_list = []
    for month_id in range(1, 13):
        for dow_id in range(7):
            for hour_id in range(24):
                month_id_list.append(month_id)
                dow_id_list.append(dow_id)
                hour_id_list.append(hour_id)

    return pd.DataFrame({'month_id': month_id_list, 'dow_id': dow_id_list, 'hour_id': hour_id_list})


def calculate_hourly_avg_contribution(*, df_hourly_sales: pd.DataFrame, date_col: str, e_d: datetime) -> pd.DataFrame:
    uuid = df_hourly_sales['unique_id'].values[0]
    s_d = df_hourly_sales['ds'].min()
    df_output = prepare_sales_for_hourly_smear_agg(df_hourly_sales=df_hourly_sales, date_col=date_col, s_d=s_d, e_d=e_d)
    df_daily_sales = df_output.groupby(by=['dt_str']) \
        .agg(**{'daily_total': pd.NamedAgg(column="y_adj", aggfunc="sum")}).reset_index()
    df_output = df_output.merge(df_daily_sales, on=['dt_str'])
    df_output['perc_daily_sales'] = df_output['y_adj'] / df_output['daily_total']
    agg_dict = {'avg_hour_contribution': pd.NamedAgg(column='perc_daily_sales', aggfunc="mean")}
    # Basic dow sales aggregates
    df_dow_hr = df_output.groupby(by=['dow_id', 'hour_id']).agg(**agg_dict).reset_index()
    df_mdh = create_month_dow_hr_df()
    df_dow_hr = df_mdh.merge(df_dow_hr, how='left', on=['dow_id', 'hour_id'])
    df_dow_hr.fillna({"avg_hour_contribution": 0}, inplace=True)
    # Monthly dow hr contributions
    df_monthly_dow_hr = df_output.groupby(by=['month_id', 'dow_id', 'hour_id']).agg(**agg_dict).reset_index()
    df_monthly_dow_hr.dropna(inplace=True)
    if df_monthly_dow_hr.shape[0] < 2016:
        # Combine monthly dow hr contrib with basic dow hr contrib
        df_left_anti = left_anti_join(df_dow_hr, df_monthly_dow_hr, ['dow_id', 'month_id', 'hour_id'])
        df_final = df_monthly_dow_hr.merge(df_left_anti, how='outer', indicator=False)
    else:
        df_final = df_monthly_dow_hr.copy()
    df_final['unique_id'] = uuid
    df_final_cols = sorted(df_final.columns)
    df_final.sort_values(by=['month_id', 'dow_id', 'hour_id'], inplace=True)
    assert df_final.shape[0] == 2016, "Incorrect number of rows"
    return df_final[df_final_cols].reset_index(drop=True).copy()


def left_anti_join(df_left: pd.DataFrame, df_right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    outer = df_left[cols].merge(df_right[cols], how='outer', indicator=True)
    anti_join = outer[outer['_merge'] == "left_only"].drop("_merge", axis=1)
    return anti_join.merge(df_left, how='left')


def _subset_order_hol_sf_columns(*, df_input: pd.DataFrame,
                                 forecast_horizon_days: int, sales_type: str) -> pd.DataFrame:
    df_output = df_input.copy()
    # Order hol sf in terms of preference we want to use.
    # Left to right - Most accurate hol SF for that business right through to country wide aggregates
    hol_sf_pref = ['branch_median', 'brand_median', 'city_median', 'country_median']
    hol_sf_pref_ = [f"sf_{forecast_horizon_days}_days_{sales_type}_{pref}" for pref in hol_sf_pref]

    # Check that all expected hol sf columns are present
    for sf_col in hol_sf_pref_:
        assert sf_col in df_output.columns, f"{sf_col} not present in input df"

    return df_output.loc[:, hol_sf_pref_]


def create_optimal_hol_sf_column(*, df_input: pd.DataFrame,
                                 forecast_horizon_days: int,
                                 sales_type: str,
                                 min_sf: float | None = None,
                                 max_sf: float | None = None,
                                 ) -> pd.DataFrame:
    if not min_sf:
        min_sf = -np.inf
    if not max_sf:
        max_sf = np.inf
    df_output = df_input.copy()
    df_subset = _subset_order_hol_sf_columns(df_input=df_input,
                                             forecast_horizon_days=forecast_horizon_days,
                                             sales_type=sales_type
                                             )
    df_output[f'sf_final_{forecast_horizon_days}_days_{sales_type}'] = (df_subset
                                                                        .replace(1, np.nan)
                                                                        .bfill(axis=1)
                                                                        .iloc[:, 0]
                                                                        .replace(np.nan, 1.0)
                                                                        .clip(lower=min_sf, upper=max_sf)
                                                                        )
    return df_output


def create_label_optimal_hol_sf_column(*, df_input: pd.DataFrame,
                                       forecast_horizon_days: int, sales_type: str) -> pd.DataFrame:
    df_output = df_input.copy()
    df_subset = _subset_order_hol_sf_columns(df_input=df_input,
                                             forecast_horizon_days=forecast_horizon_days,
                                             sales_type=sales_type
                                             )
    # Set placeholder column to highlight when no hol sf exists
    cols = list(df_subset.columns)
    placeholder_col = f'sf_{forecast_horizon_days}_days_{sales_type}_none'
    cols += [placeholder_col]
    df_subset[placeholder_col] = 0.9
    df_output[f'sf_final_{forecast_horizon_days}_days_{sales_type}_label'] = (df_subset[cols]
                                                                              .replace(1, np.nan)
                                                                              .notna()
                                                                              .idxmax(axis=1)
                                                                              )
    return df_output
