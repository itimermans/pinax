import numpy as np
import pandas as pd


def vspy_buffer_csv_to_df(file_path):
    """Takes csv auto-generated from vspy buffer binary (vsb) conversion and returns the
    clean table as a dataframe.

    """
    # Parameters
    table_marker = "Line"
    target_header_num = 3  # Should be 1 or more
    total_headers_num = 4
    limit_lines_search = 300

    # Loop
    flag_marker = False
    with open(file_path) as f:
        for i, line in enumerate(f):
            if (table_marker in line) and not flag_marker:
                # Assignation
                target_header_row = i + target_header_num - 1
                table_row = i + total_headers_num
                flag_marker = True

            if flag_marker:
                if i == target_header_row:
                    # Header row extraction
                    columns = [col.strip() for col in line.split(",")]
                    break

            if i > limit_lines_search:
                raise ValueError(
                    f"Marker '{table_marker}' not found within the first "
                    f"{limit_lines_search} lines of the file."
                )

    # DataFrame
    df = pd.read_csv(file_path, skiprows=table_row, header=None, names=columns)
    return df


def vspy_buffer_df_extend_bytes(df):
    # Efficient, vectorized payload extraction for up to 64 bytes

    # Parameters
    extended_payload_cols = [f"B{i}" for i in range(1, 65)]
    current_payload_cols = [f"B{i}" for i in range(1, 9)]

    # Identify rows with long payloads (space in B1)
    long_payload_mask = df["B1"].astype(str).str.contains(" ")

    # If the mask is empty return the original df
    if not long_payload_mask.any():
        print("No extended payload detected, returnig original dataframe")
        return df

    # Split long payloads into bytes (vectorized) and rename columns to B1, B2, ..., B64
    long_payload_bytes = (
        df.loc[long_payload_mask, "B1"].str.strip().str.split(" ", expand=True)
    )
    long_payload_bytes.columns = extended_payload_cols[: long_payload_bytes.shape[1]]

    # For short payloads, stack B1-B8 columns as strings (vectorized)
    short_payload_bytes = df.loc[~long_payload_mask, current_payload_cols]

    # Pad both to 64 columns
    def pad_to_64(df_bytes):
        cols = [f"B{i}" for i in range(1, 65)]
        df_bytes = df_bytes.reindex(columns=cols, fill_value=np.nan)
        return df_bytes

    long_payload_bytes = pad_to_64(long_payload_bytes)
    short_payload_bytes = pad_to_64(short_payload_bytes)

    # Combine back into one DataFrame, preserving original order
    payload_df = pd.concat([long_payload_bytes, short_payload_bytes]).sort_index()

    # Reorganize columns to match the original payload structure
    first_payload_idx = df.columns.get_loc(current_payload_cols[0])
    new_col_order = (
        list(df.columns[:first_payload_idx])
        + extended_payload_cols
        + list(df.columns[first_payload_idx + len(current_payload_cols) :])
    )
    # Create the extended dataframe
    df_extended = pd.concat([df.drop(columns=current_payload_cols), payload_df], axis=1)
    df_extended = df_extended[new_col_order]

    return df_extended
