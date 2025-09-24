import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .tdms_tools import TDMSFinder, df_from_fullTDMS


class DataLoader:
    def __init__(self, path):
        self.finder = TDMSFinder(path)

    def load_test_df(self, id):
        filepath = self.finder.find_tdms_path(id)
        df = df_from_fullTDMS(filepath)[1:].reset_index(drop=True)
        return df

    def load_processed_test(self, id):
        df = self.load_test_df(id)
        df = dyno_signals(id, df)
        df = jeep_signals(id, df)
        return df

    def load_vspy(self, id, start_string="VSpy"):
        folder = os.path.dirname(self.finder.find_tdms_path(id))
        files = [f for f in os.listdir(folder) if f.startswith(start_string)]
        if len(files) == 0:
            raise ValueError(f"No files found starting with {start_string}")
        elif len(files) > 1:
            print(f"Multiple files found starting with {start_string}:")
            for f in files:
                print(f)
            raise ValueError(
                f"Multiple files found starting with {start_string}. Please specify"
                f"the correct file."
            )
        vspy_file = files[0]
        if vspy_file.lower().endswith(".csv"):
            return pd.read_csv(os.path.join(folder, vspy_file), low_memory=False)
        else:
            raise ValueError(f"Found file {vspy_file} is not a CSV file.")

    def load_processed_vspy(self, id):
        df = self.load_vspy(id)
        df = process_vspy(df)
        return df

    def load_full_processed_test(self, id):
        df = self.load_processed_test(id)
        vspy = self.load_processed_vspy(id)
        df = replace_vspy_columns(df, vspy)
        df = sanitize_df(df)
        return df


# Aux Functions


# Global Tests Data
tests_bags_grade = {
    "62406020": {1: 0, 2: 0, 3: 3, 4: 6},
    "62406029": {1: 0, 2: 0, 3: 3, 4: 6},
    "62406033": {1: 0, 2: 0, 3: 3, 4: 6},
    "62406024": {1: 25},
    # Etransit
    "62210043": {1: 0, 2: 1, 3: 3},
}


def dyno_signals(id, test):
    # Physical magnitudes conversion factors
    conv_lbf_to_N = 4.44822
    conv_lb_to_kg = 0.453592
    conv_mph_to_mps = 0.44704
    conv_kg_to_N = 9.81

    # Physical signals
    test["calc_Accel_mps2"] = pd.DataFrame(
        np.gradient(test["Dyno_Spd[mph]"] * conv_mph_to_mps, test["Time[s]"])
    )
    test["calc_Accel_mps2_filtered"] = (
        pd.DataFrame(
            np.gradient(test["Dyno_Spd[mph]"] * conv_mph_to_mps, test["Time[s]"])
        )
        .rolling(window=3)
        .mean()
        .rolling(window=3)
        .mean()
    )

    # Categoricals
    test["cat_Exhaust_Bag"] = pd.Categorical(test["Exhaust_Bag"].round().astype(int))
    test["cat_Test_ID"] = pd.Categorical(test["Test_ID"].round().astype(int))
    test["cat_Test_and_Bag"] = pd.Categorical(
        test["cat_Test_ID"].astype(str) + "-" + test["cat_Exhaust_Bag"].astype(str)
    )

    # Grade Signal (using cat_Exhaust_Bag but it can also use Exhaust_Bag,
    # just in case rounding to int:
    # test['test_Grade_perc'] = test['Exhaust_Bag'].round().astype(int).map(lambda x:
    # tests_bags_grade.get(id,{}).get(x,0)).plot()
    test["test_Grade_perc"] = test["cat_Exhaust_Bag"].map(
        lambda x: tests_bags_grade.get(id, {}).get(x, 0)
    )

    # Tractive force and power calculations

    # Force from grade
    test["calc_TractiveForce_Grade_N"] = (
        (pd.to_numeric(test.attrs["TestInfo_Dyno_Test_Weight_lb"]) * conv_lb_to_kg)
        * conv_kg_to_N
        * np.sin(np.arctan(test["test_Grade_perc"].astype(int) / 100))
    )

    # Tractive power using the dyno-generated signal Dyno_TractiveForce[N]
    test["calc_Dyno_TractivePower_W"] = (
        test["Dyno_Spd[mph]"] * conv_mph_to_mps * test["Dyno_TractiveForce[N]"]
    )

    # Tractive force using the 'set' coefficients...first static (just coefficients,
    # no acceleration factor), then add the effects of acceleration and grade (should
    # be equal to Dyno_TractiveForce[N]), then power
    test["calc_TractiveForceSet_Static_N"] = (
        pd.to_numeric(test.attrs["TestInfo_Dyno_Set_A"])
        + pd.to_numeric(test.attrs["TestInfo_Dyno_Set_B"]) * test["Dyno_Spd[mph]"]
        + pd.to_numeric(test.attrs["TestInfo_Dyno_Set_C"]) * test["Dyno_Spd[mph]"] ** 2
    ) * conv_lbf_to_N
    test["calc_TractiveForceSet_N"] = (
        test["calc_TractiveForceSet_Static_N"]
        + (pd.to_numeric(test.attrs["TestInfo_Dyno_Test_Weight_lb"]) * conv_lb_to_kg)
        * test["calc_Accel_mps2_filtered"]
        + test["calc_TractiveForce_Grade_N"]
    )
    test["calc_TractivePowerSet_W"] = (
        test["Dyno_Spd[mph]"] * conv_mph_to_mps * test["calc_TractiveForceSet_N"]
    )

    # Tractive force using the 'target' coefficients...first static (just coefficients,
    # no acceleration factor), then add the effects of acceleration and grade, then
    # power
    test["calc_TractiveForceTarget_Static_N"] = (
        pd.to_numeric(test.attrs["TestInfo_Dyno_Target_A"])
        + pd.to_numeric(test.attrs["TestInfo_Dyno_Target_B"]) * test["Dyno_Spd[mph]"]
        + pd.to_numeric(test.attrs["TestInfo_Dyno_Target_C"])
        * test["Dyno_Spd[mph]"] ** 2
    ) * conv_lbf_to_N
    test["calc_TractiveForceTarget_N"] = (
        test["calc_TractiveForceTarget_Static_N"]
        + (pd.to_numeric(test.attrs["TestInfo_Dyno_Test_Weight_lb"]) * conv_lb_to_kg)
        * test["calc_Accel_mps2_filtered"]
        + test["calc_TractiveForce_Grade_N"]
    )
    test["calc_TractivePowerTarget_W"] = (
        test["Dyno_Spd[mph]"] * conv_mph_to_mps * test["calc_TractiveForceTarget_N"]
    )

    return test


def jeep_signals(id, test):

    test["Imotors_calc"] = (
        test["Iinverter_UDF5"] - test["Icabinhtr_Idc6"] - test["Ihvac_UDF6"]
    )
    test["IHmotors_calc"] = (
        test["IHinverter_UDF7"] - test["Ihcabinhtr_Ih6"] - test["IHhvac_UDF8"]
    )
    test["Pmotors_calc"] = (
        test["Pinverter_UDF9"] - test["Pcabinhtr_P6"] - test["Phvac_UDF10"]
    )
    test["WPmotors_calc"] = (
        test["WPinverter_UDF11"] - test["WPcabinhtr_WP6"] - test["WPhvac_UDF12"]
    )

    return test


def remove_units(series, print_errors=False):
    """
    Removes units from a pandas Series by stripping everything after the first
    whitespace, then attempts to convert the entire column to numeric.
    If any conversion fails, restores the original column and prints one message.
    Uses vectorized operations for performance.
    """
    base = series.astype(str).str.split().str[0]
    try:
        # Try to convert the entire column
        return pd.to_numeric(base, errors="raise")
    except Exception:
        if print_errors:
            print(f"Unit strip failed for '{series.name}'. Restoring original values.")
        return series


def remove_units_from_columns(df, columns=None):
    """
    Applies remove_units to specified columns (or all columns if none specified).
    """
    cols = columns if columns is not None else df.columns
    for col in cols:
        if df[col].dtype == "object":
            df[col] = remove_units(df[col])
    return df


def replace_vspy_columns(df, vspy_log, ref_column="DAQ_time__s", replace_columns=None):

    df_new = df.copy()
    # Align log to df using merge_asof (nearest match)
    aligned_log = pd.merge_asof(
        df_new[[ref_column]],
        vspy_log.sort_values(ref_column),
        on=ref_column,
        direction="nearest",
        tolerance=0.05,  # adjust tolerance as needed
    )

    # TODO
    # CRITERIA TO CHANGE: CAN BE WHATEVER, RIGHT NOW __state
    common_cols = set(df.columns) & set(vspy_log.columns)
    replace_columns = df.columns if replace_columns is None else replace_columns
    # Replace values in df for selected columns
    selected_cols = [col for col in common_cols if col.endswith("__state")]
    for col in selected_cols:
        df_new[col] = aligned_log[col].where(aligned_log[col].notna(), df_new[col])
    return df_new

    # # Minimal example: aligning two time series DataFrames with merge_asof
    # import pandas as pd
    # df = pd.DataFrame({
    #     'Time': [1.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    #     'A': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    #     'C': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # })
    # log = pd.DataFrame({
    #     'Time': [1.05, 2.05, 3.05, 4.05, 5.05],
    #     'A': [100, 101, 102, 103, 104],
    #     'B': [1, 2, 3, 4, 5],
    #     'D': [10, 11, 12, 13, 14]
    # })

    # df_sorted = df.sort_values('Time')
    # log_sorted = log.sort_values('Time')

    # aligned_log = pd.merge_asof(
    #     df_sorted[['Time']],
    #     log_sorted,
    #     on='Time',
    #     direction='nearest',
    #     tolerance=0.1
    #     # Only rows in log within 0.1 of df's Time will be matched
    # )

    # print('df_sorted:')
    # print(df_sorted)
    # print('\nlog_sorted:')
    # print(log_sorted)
    # print('\naligned_log:')
    # print(aligned_log)


def process_vspy(df):
    # Columns TCM_Current_Gear__state and TCM_Target_Gear__state as text, but the rest
    # the same
    try:
        df["TCM_Current_Gear__state"] = df["TCM_Current_Gear__state"].astype(str)
        df["TCM_Target_Gear__state"] = df["TCM_Target_Gear__state"].astype(str)
    except Exception:
        pass

    # REMOVE UNITS: In all columns, look for spaces and remove them and anything after
    # them
    df = remove_units_from_columns(df)

    return df


def sanitize_df(df):
    # TYPE SANITATION: object columns to str
    # TODO: Clean this up
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")  # pandas' string dtype

    return df


def estimate_grade(
    df,
    signal_speed_mph,
    signal_time_s,
    signal_dyno_TractiveForce_N,
    attr_Set_A,
    attr_Set_B,
    attr_Set_C,
    attr_testWeight_lb,
    signal_Exhaust_Bag=None,
    print_results=False,
    plot_figure=False,
):

    # Physical magnitudes conversion factors
    conv_lbf_to_N = 4.44822
    conv_lb_to_kg = 0.453592
    conv_mph_to_mps = 0.44704
    conv_kg_to_N = 9.81

    # Work on a copy
    _df = df.copy()

    # Acceleration from speed
    _df["calc_Accel_mps2"] = pd.DataFrame(
        np.gradient(_df[signal_speed_mph] * conv_mph_to_mps, _df[signal_time_s])
    )
    # Filter Acceleration, double window
    _df["calc_Accel_mps2_filtered"] = (
        _df["calc_Accel_mps2"].rolling(window=5).mean().rolling(window=5).mean()
    )

    # On one side:
    # Calculate Dyno Set Static Tractive Force with Set coefficients ...
    _df["calc_TractiveForceSet_Static_N"] = (
        pd.to_numeric(_df.attrs[attr_Set_A])
        + pd.to_numeric(_df.attrs[attr_Set_B]) * _df[signal_speed_mph]
        + pd.to_numeric(_df.attrs[attr_Set_C]) * _df[signal_speed_mph] ** 2
    ) * conv_lbf_to_N
    # ... and add Mass*Acceleration to get Set Tractive Force
    _df["calc_TractiveForceSet_N"] = (
        _df["calc_TractiveForceSet_Static_N"]
        + (pd.to_numeric(_df.attrs[attr_testWeight_lb]) * conv_lb_to_kg)
        * _df["calc_Accel_mps2_filtered"]
    )

    # On the other side: Dyno Tractive Force, which is ready as
    # signal_dyno_TractiveForce_N

    # Compare signals. The only difference should be the grade
    # Force from grade
    _df["calc_est_grade_N"] = (
        _df[signal_dyno_TractiveForce_N] - _df["calc_TractiveForceSet_N"]
    )
    # As a percentage
    _df["calc_est_grade_percent"] = 100 * (
        np.tan(
            np.asin(
                _df["calc_est_grade_N"]
                / (
                    (pd.to_numeric(_df.attrs[attr_testWeight_lb]) * conv_lb_to_kg)
                    * conv_kg_to_N
                )
            )
        )
    )

    # Heavily Filtered for some easier result visualization
    _df["calc_est_grade_percent_filtered"] = (
        _df["calc_est_grade_percent"].rolling(window=20).mean()
    )

    # Plot if requested
    # Plot filtered grade with speed
    if plot_figure:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=_df[signal_time_s],
                    y=_df["calc_est_grade_percent"],
                    name="Estimated Grade [%]",
                    mode="lines",
                    line=dict(color="red"),
                    yaxis="y",
                ),
                go.Scatter(
                    x=_df[signal_time_s],
                    y=_df[signal_speed_mph],
                    name="Speed [mph]",
                    mode="lines",
                    line=dict(color="grey"),
                    yaxis="y2",
                ),
            ]
            + (
                [
                    go.Scatter(
                        x=_df[signal_time_s],
                        y=_df[signal_Exhaust_Bag],
                        name="Exhaust Bag",
                        mode="lines",
                        line=dict(color="orange"),
                        yaxis="y",
                    )
                ]
                if signal_Exhaust_Bag is not None
                else []
            ),
            layout=go.Layout(
                title=_df.attrs.get("TestInfo_Test_ID", "Test ") + "  Grade Estimation",
                xaxis=dict(title="Time [s]"),
                # Main y axis
                yaxis=dict(title="Estimated Grade [%]"),
                # Secondary y-axis
                yaxis2=dict(
                    title="Speed [mph]", overlaying="y", side="right", showgrid=False
                ),
                legend=dict(
                    xanchor="left", yanchor="bottom", x=1, y=0, orientation="v"
                ),
            ),
        )
        fig.show()

    # Print if requested
    # If Exhaust Bag is provided, we clean it (there are interpolated values)
    # and group by bag
    if print_results:
        if signal_Exhaust_Bag is not None:
            # Clean Exhaust Bag
            _df["clean_Exhaust_Bag"] = _df[signal_Exhaust_Bag].round(0)

            _table_byBag = (
                _df.groupby("clean_Exhaust_Bag")[["calc_est_grade_percent"]]
                .agg(["mean", lambda x: np.std(x) / np.mean(x)])
                .round(2)
            )
            _table_byBag.columns = ["mean", "coef_var"]
            print("Grade estimation by Exhaust Bag (Mean and Coefficient of Variation)")
            print(_table_byBag)

        else:
            print("Provide signal_Exhaust_Bag signal to get grade estimation by bag")
            print(
                "Mean and Coefficient of Variation of grade estimation for the full "
                "test is:"
            )
            print(
                _df["calc_est_grade_percent"].mean().round(2),
                _df["calc_est_grade_percent"].std().round(2),
            )

    # Return calc_est_grade_percent
    return _df["calc_est_grade_percent"]


def bulk_load_parquet_data(testlist, data_dir="./parquet_data/"):
    data = dict()

    for test in testlist:
        file_path = os.path.join(data_dir, f"{test}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            data[str(test)] = df
        else:
            print(f"File {file_path} does not exist.")
    return data


def tests_to_single_df(tests):
    # Test preparation
    #   - Pure dataframe result of doing pd.concat([test_df_1,test_df_2,...]) OK ->
    # This is the "standard" format. All others are converted to this
    #   - Dictionary with format {test_id: test_df} OK
    #   - List of test_df OK

    if type(tests) is pd.DataFrame:
        # Check if format is ok?
        tests_df = tests
    elif type(tests) is dict:
        tests_df = pd.concat([test for test in tests.values()], axis=0)
    elif type(tests) is list:
        tests_df = pd.concat(tests, axis=0)

    return tests_df.reset_index(drop=True)


def table_categories(
    tests,
    categories=["Test_ID", "Exhaust_Bag"],
    group_level="Exhaust_Bag",
    float_to_int=True,
    remove_values={"Exhaust_Bag": 0},
):
    """
    Processes test data and returns a table (DataFrame) where each row represents a
    unique combination of categories.

    Example: table_categories(tests = df, categories = ['Test_ID', 'Exhaust_Bag'],
    group_level = 'Exhaust_Bag',float_to_int = True, remove_values =
    {'Exhaust_Bag' : 0})

    Will return rows with unique bags for each Test ID

    Args:
        tests (pd.DataFrame, dict, or list): The test data to process. Can be a
        single DataFrame,
                                                a dictionary of DataFrames with
                                                test IDs as keys,
                                                or a list of DataFrames.
        categories (list, optional): List of column names to use as categories.
        Defaults to ['Test_ID', 'Exhaust_Bag'].
        group_level (str, optional): The column name to use as the grouping level.
        Defaults to 'Exhaust_Bag'.
        float_to_int (bool, optional): If True, converts float columns in categories to
        integers. Defaults to True.
        remove_values (dict, optional): Dictionary specifying values to remove
        from the categories.
                                        Keys must be in categories, and values
                                        must be lists or single numbers.
                                        Defaults to {'Exhaust_Bag': 0}.
    Returns:
        pd.DataFrame: A DataFrame where each row represents a unique
        combination of categories
    Raises:
        ValueError: If a category is not a valid column in the DataFrame.
        ValueError: If keys in remove_values are not in categories.
        ValueError: If values in remove_values are not lists or single numbers.
        ValueError: If group_level is not in categories.
    """

    # Tests to single dataframe
    tests_df = tests_to_single_df(tests)

    # Check if categories are valid columns in the dataframe
    for cat in categories:
        if cat not in tests_df.columns:
            raise ValueError(f"Category '{cat}' is not a valid column in the dataframe")

    # Extract the categories columns and round floats to int if indicated
    categories_df = tests_df[categories]
    if float_to_int:
        categories_df = categories_df.apply(
            lambda x: x.astype(int) if x.dtype == float else x
        )

    # Get uniques
    bags_df_uniques = (
        categories_df.groupby(categories, as_index=False, sort=False)
        .first()
        .reset_index(drop=True)
    )

    # Remove values:
    # Check args: They keys must be in the categories
    if not all([key in categories for key in remove_values.keys()]):
        raise ValueError("Keys in remove_values must be in categories")
    # Values must be lists. If a value is a single number, convert it to a list
    for key, values in remove_values.items():
        if not isinstance(values, list):
            if isinstance(values, (int, float)):
                remove_values[key] = [values]
            else:
                raise ValueError(
                    f"Values for key '{key}' must be a list or a single number"
                )

    #  Take the arg remove_values and remove the rows with the values indicated for
    # the columns indicated in the keys
    for key, values in remove_values.items():
        bags_df_uniques = bags_df_uniques[~bags_df_uniques[key].isin(values)]

    # Group
    # If group_level is in categories and is not the last, group by all categories up
    # to that one, aggregating as list
    if group_level in categories:
        if group_level != categories[-1]:
            bags_df_grouped = (
                bags_df_uniques.groupby(
                    categories[: categories.index(group_level) + 1],
                    as_index=False,
                    sort=False,
                )
                .agg(list)
                .reset_index(drop=True)
            )
        else:
            bags_df_grouped = bags_df_uniques
    else:
        raise ValueError(f"Group level '{group_level}' is not in categories")

    return bags_df_grouped

    # # Try
    # a = table_categories(tests, categories = ['Test_ID', 'Exhaust_Bag'],
    # group_level='Test_ID')
    # a


def table_functions(tests, table_categories, functions):

    # Validate inputs:
    # Check that the table_categories is a DataFrame
    if not isinstance(table_categories, pd.DataFrame):
        raise ValueError(
            "table_categories must be a DataFrame, see table_categories function"
        )
    # Check that functions is a dictionary with functions as values
    if not isinstance(functions, dict):
        raise ValueError(
            "functions must be a dictionary with names (strings) as keys and functions "
            "as values"
        )
    if not all([callable(fun) for fun in functions.values()]):
        raise ValueError(
            "All values in functions must be callable functions. Example: {'Mean': np."
            "mean, 'Final time': lambda df : df['Time[s]'].iloc[-1]} "
        )

    # Tests to single dataframe
    tests_df = tests_to_single_df(tests)

    # Initialize results DataFrame with copy of table
    results = table_categories.copy()

    # Iterate over rows in table
    for i, row in table_categories.iterrows():
        # Start with True mask and additively filter per each category
        mask = True
        for cat, value in row.items():
            # Value can be a value or a list. Standardize to list
            value = [value] if not isinstance(value, list) else value
            mask = mask & (tests_df[cat].isin(value))
        # Check that the mask is not empty (it would generate an empty df)
        if mask.empty:
            raise ValueError(
                f"Empty result for category table row {i}, with values {row}"
            )
        # Result of filtering is df_row_filtered, one different for each row
        df_row_filtered = tests_df[mask]

        # Apply functions
        for fun_name, fun in functions.items():
            results.loc[i, fun_name] = fun(df_row_filtered)

    return results

    # # Try

    # table_cats = table_categories(tests, categories = ['Test_ID', 'Exhaust_Bag'],
    # group_level='Test_ID')
    # functions = {
    #     'Torque Mean': lambda df: np.mean(df['Dyno_Spd[mph]']),
    #     'Combined Motor Currents Mean Calc': lambda df:
    # np.mean(df['calc_I_motorComb']),
    #     'Combined Motor Currents Mean': lambda df:
    # np.mean(df['I_mfront']+df['I_mrear'])
    # }
    # table_funs = table_functions(tests,table_cats,functions = functions)
    # table_funs
