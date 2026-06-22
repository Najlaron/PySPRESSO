import ast
import re

import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState


@register_operation(
    id="loader_data",
    label="Load Data",
    description="Load a data matrix from a CSV file.",
    citation="",
    category_tags=[OperationTag.IO, OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef(
            name="data_input_file_name",
            type="str",
            required=True,
            label="Data file",
            help="Path to the input CSV file.",
        ),
        ParameterDef(
            name="separator",
            type="str",
            required=False,
            default=";",
            label="Separator",
        ),
        ParameterDef(
            name="encoding",
            type="str",
            required=False,
            default="ISO-8859-1",
            label="Encoding",
        ),
    ],
    requires=[],
    produces=["data"],
)
def loader_data(
    state: WorkflowState,
    data_input_file_name: str,
    separator: str = ";",
    encoding: str = "ISO-8859-1",
):
    state.data = pd.read_csv(
        data_input_file_name,
        sep=separator,
        encoding=encoding,
    )

    return {
        "rows": int(state.data.shape[0]),
        "columns": int(state.data.shape[1]),
    }


@register_operation(
    id="add_cpdID",
    label="Add cpdID",
    description="Create cpdID from m/z and retention time columns.",
    citation="",
    category_tags=[OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef("mz_col", "str", False, "m/z", "m/z column"),
        ParameterDef("rt_col", "str", False, "RT [min]", "RT column"),
        ParameterDef("round_mz_col", "int", False, 5, "m/z decimals"),
        ParameterDef("round_rt_col", "int", False, 3, "RT decimals"),
    ],
    requires=["data"],
    produces=["data"],
)
def add_cpdID(
    state: WorkflowState,
    mz_col: str = "m/z",
    rt_col: str = "RT [min]",
    round_mz_col: int = 5,
    round_rt_col: int = 3,
):
    data = state.data

    if data is None:
        raise ValueError("No data loaded.")

    if mz_col not in data.columns:
        raise ValueError(f"Column not found: {mz_col}")

    if rt_col not in data.columns:
        raise ValueError(f"Column not found: {rt_col}")

    data["cpdID"] = (
        "M"
        + data[mz_col].round(round_mz_col).astype(float).astype(str)
        + "-T"
        + data[rt_col].round(round_rt_col).astype(float).astype(str)
    )

    data["Duplicate_Count"] = data.groupby("cpdID").cumcount()
    is_duplicate = data["Duplicate_Count"] > 0
    data.loc[is_duplicate, "cpdID"] = (
        data.loc[is_duplicate, "cpdID"]
        + "_"
        + data.loc[is_duplicate, "Duplicate_Count"].astype(str)
    )
    data.drop("Duplicate_Count", axis=1, inplace=True)

    state.data = data

    return {
        "rows": int(data.shape[0]),
        "cpdID_added": True,
    }


@register_operation(
    id="add_cpdID_from_column",
    label="Add cpdID from column",
    description="Create cpdID from an existing column.",
    citation="",
    category_tags=[OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef(
            name="cpdID_col",
            type="str",
            required=True,
            label="cpdID column",
            help="Column to use as cpdID, for example Compound Name, Formula, or another identifier.",
        ),
    ],
    requires=["data"],
    produces=["data"],
)
def add_cpdID_from_column(
    state: WorkflowState,
    cpdID_col: str,
):
    data = state.data

    if data is None:
        raise ValueError("No data loaded.")

    if cpdID_col not in data.columns:
        raise ValueError(f"Column not found: {cpdID_col}")

    data["cpdID"] = data[cpdID_col].astype("string").str.replace(";", ",,", regex=False)

    state.data = data

    return {
        "rows": int(data.shape[0]),
        "cpdID_added": True,
        "source_column": cpdID_col,
        "unique_cpdID": int(data["cpdID"].nunique(dropna=True)),
        "duplicate_cpdID": int(data["cpdID"].duplicated().sum()),
    }


@register_operation(
    id="extracter_variable_metadata",
    label="Extract Variable Metadata",
    description="Extract variable metadata columns from the loaded data.",
    citation="",
    category_tags=[OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef(
            name="columns_by_name",
            type="list",
            required=False,
            default=["cpdID", "Name", "Formula"],
            label="Columns by name",
            help="Named columns to keep in variable metadata.",
        ),
        ParameterDef(
            name="column_index_ranges",
            type="list",
            required=False,
            default=[(10, 15), (18, 23)],
            label="Column index ranges",
            help="Column index ranges or individual column indexes to include, e.g. [(10, 15), (18, 23)] or [10, 12, (18, 23)].",
        ),
    ],
    requires=["data"],
    produces=["variable_metadata"],
)
def extracter_variable_metadata(
    state: WorkflowState,
    columns_by_name=["cpdID", "Name", "Formula"],
    column_index_ranges=[(10, 15), (18, 23)],
):
    data = state.data

    if data is None:
        raise ValueError("No data loaded.")

    columns_by_name = _parse_list_like(columns_by_name)
    column_index_ranges = _parse_list_like(column_index_ranges)

    if columns_by_name is None:
        columns_by_name = []

    if column_index_ranges is None:
        column_index_ranges = []

    missing_columns = [col for col in columns_by_name if col not in data.columns]

    if missing_columns:
        raise ValueError("Column(s) not found: " + ", ".join(missing_columns))

    variable_metadata = data.loc[:, columns_by_name].copy()

    for column_index_range in column_index_ranges:

        # Single column index, e.g. 10
        if not isinstance(column_index_range, tuple):
            column_index = int(column_index_range)

            if column_index < 0 or column_index >= data.shape[1]:
                raise ValueError(f"Column index {column_index} is out of bounds.")

            column_name = data.columns[column_index]

            if column_name in variable_metadata.columns:
                print(
                    f"Column {column_index} is already included. "
                    "Skipping this column."
                )
            else:
                variable_metadata = variable_metadata.join(data.iloc[:, column_index])

        # Column range, e.g. (10, 15)
        else:
            start_index = int(column_index_range[0])
            end_index = int(column_index_range[1])

            if start_index < 0 or end_index > data.shape[1]:
                raise ValueError(
                    f"Column index range {column_index_range} is out of bounds."
                )

            selected_columns = list(data.columns[start_index:end_index])

            cols_to_add = [
                col for col in selected_columns if col not in variable_metadata.columns
            ]

            skipped_columns = [
                col for col in selected_columns if col in variable_metadata.columns
            ]

            if skipped_columns:
                print(
                    f"Some columns in the range {column_index_range} "
                    "are already included. Including only the new columns."
                )

            if cols_to_add:
                variable_metadata = variable_metadata.join(data[cols_to_add])

    state.variable_metadata = variable_metadata

    return {
        "rows": int(variable_metadata.shape[0]),
        "columns": int(variable_metadata.shape[1]),
        "variable_metadata_created": True,
        "included_columns": list(variable_metadata.columns),
    }


@register_operation(
    id="extracter_data",
    label="Extract Data Matrix",
    description="Extract the cpdID column and intensity columns matching a selected prefix.",
    citation="",
    category_tags=[OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef(
            name="prefix",
            type="str",
            required=False,
            default="Area:",
            label="Intensity column prefix",
            help="Prefix used to select sample intensity columns, for example 'Area:'.",
        ),
    ],
    requires=["data"],
    produces=["data"],
)
def extracter_data(
    state: WorkflowState,
    prefix: str = "Area:",
):
    data = state.data

    if data is None:
        raise ValueError("No data loaded.")

    if "cpdID" not in data.columns:
        raise ValueError(
            "Column not found: cpdID. Run add_cpdID or add_cpdID_from_column first."
        )

    temp_data = data.copy()

    area_columns = temp_data.filter(
        regex=r"^" + re.escape(prefix),
        axis=1,
    )

    if area_columns.shape[1] == 0:
        raise ValueError(f"No columns found starting with prefix: {prefix}")

    extracted_data = temp_data[["cpdID"]].join(area_columns)

    # Replace NaN with 0.
    # These zeros can later be treated as missing values.
    extracted_data = extracted_data.fillna(0)

    state.data = extracted_data

    return {
        "rows": int(extracted_data.shape[0]),
        "columns": int(extracted_data.shape[1]),
        "data_matrix_created": True,
        "prefix": prefix,
        "intensity_columns": int(area_columns.shape[1]),
    }


## HELP FUNCTIONS ---------------------------


def _parse_list_like(value):
    """
    Accept either a real Python list/tuple or a string representation of one.

    Examples
    --------
    ["cpdID", "Name"]
    "['cpdID', 'Name']"
    [(10, 15), (18, 23)]
    "[(10, 15), (18, 23)]"
    """

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Could not parse value as a Python list/tuple: {value}")

    return value
