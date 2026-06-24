import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState


@register_operation(
    id="loader_data",
    label="Load Data",
    description="Load a data matrix from a CSV file.",
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
