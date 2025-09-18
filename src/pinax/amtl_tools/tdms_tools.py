import os
from collections import OrderedDict

import pandas as pd
from nptdms import TdmsFile


# %% TDMSFinder
# Specific function that maps a directory, targets one that starts with a given string,
# and looks for a .tdms file inside that directory that also starts with the same string
# TODO: Generalize this function to a "TDMSFinder" class that works on many structures
class TDMSFinder:
    def __init__(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"{directory} is not a valid directory.")
        self.directory = directory
        # Cache all folders in the directory
        self.folders = [
            f
            for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))
        ]

    def find_tdms_path(self, start_string):
        # Ensure start_string is a string
        start_string = str(start_string)

        # Search cached folders for match
        for folder in self.folders:
            if folder.startswith(start_string):
                folder_path = os.path.join(self.directory, folder)
                # Search for .tdms file inside folder
                for file_item in os.listdir(folder_path):
                    if file_item.startswith(start_string) and file_item.endswith(
                        ".tdms"
                    ):
                        file_path = os.path.join(folder_path, file_item)
                        if os.path.isfile(file_path):
                            return file_path
                        else:
                            print(
                                f"File {file_item} located with errors at {file_path}"
                            )
                            return file_path

                print(
                    (
                        f"Folder starting with '{start_string}' found, but no matching "
                        f"'.tdms' file inside."
                    )
                )
                return None

        print(
            f"No folder starting with '{start_string}' containing"
            f" a file ending with '.tdms' found."
        )
        return None


# %% TDMS Conversion Tools


# df_from_fullTDMS: Opens a full TDMS file (from dyno testing) and returns a dataframe
# with the data and properties
def df_from_fullTDMS(full_TDMS_path: str):

    # Channels from Data
    try:
        fullTDMS = TdmsFile(full_TDMS_path)
    except FileNotFoundError:
        print(
            f"FileNotFoundError at {full_TDMS_path}\nFile could be corrupt "
            f"(sometimes happens copying TDMS files) or path can be incorrect"
        )
        return None

    df_Data = fullTDMS["Data"].as_dataframe()

    # Properties

    # Function to correct the keys..remove certain characters
    # and replace others with underscores

    def correct_keys(keys: list):

        replace_chars = ["+", "-", "*", "/", " "]
        remove_chars = ["[", "]", "#"]

        replace_remove_table = str.maketrans(
            "".join(replace_chars), "_" * len(replace_chars), "".join(remove_chars)
        )
        keys = keys.translate(replace_remove_table)

        # Remove underscores at the end of the string, one or many
        keys = keys.rstrip("_")

        return keys

    # Empty OrderedDict to store properties
    properties = OrderedDict()

    # description ...empty?
    properties["description"] = ""

    # DateTime
    properties["DateTime"] = fullTDMS["TestInfo"].properties["Test Start Time"]

    # Properties from PhaseData, adding "Phase_"
    properties.update(
        {
            correct_keys("Phase_" + c.name + "_" + p): (
                c.properties[p] if pd.notnull(c.properties[p]) else ""
            )
            for c in fullTDMS["PhaseData"].channels()
            for p in c.properties
        }
    )

    # Properties from Data
    properties.update(
        {correct_keys(key): value for key, value in fullTDMS["Data"].properties.items()}
    )

    # Properties from TestInfo to the OrderedDict, adding "TestInfo_"
    properties.update(
        {
            correct_keys("TestInfo_" + key): value
            for key, value in fullTDMS["TestInfo"].properties.items()
        }
    )

    # Add the properties to the dataframe
    df_Data.attrs = properties

    # Return df_Data
    return df_Data
