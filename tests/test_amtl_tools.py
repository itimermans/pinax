import os

from nptdms import TdmsFile

from pinax.amtl_tools import TDMSFinder, df_from_fullTDMS

EXAMPLES_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "data")
ID = 69909009
TDMS_DATA_PATH = os.path.join(
    EXAMPLES_DATA_DIR, "69909009 Test Data Folder", "69909009 Test Data.tdms"
)


def test_TDMSFinder():
    finder = TDMSFinder(EXAMPLES_DATA_DIR)
    path = finder.find_tdms_path(str(ID))
    assert path is not None
    assert path.endswith(".tdms")

    tdms_file = TdmsFile(path)
    assert tdms_file is not None


def test_df_from_fullTDMS():
    print(TDMS_DATA_PATH)
    df = df_from_fullTDMS(TDMS_DATA_PATH)
    assert df is not None
