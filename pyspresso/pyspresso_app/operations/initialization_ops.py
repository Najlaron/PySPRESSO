from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.registry import register_operation
from core.operation_models import OperationTag, ParameterDef
from core.workflow_models import WorkflowState
from core.pdf_reporter import Report


@register_operation(
    id="initializer_compound_discoverer",
    label="Initialize Compound Discoverer Dataset",
    description=(
        "Load Compound Discoverer data and batch info, create cpdID, "
        "extract variable metadata and intensity matrix, reorder samples, "
        "and create metadata."
    ),
    citation="",
    category_tags=[OperationTag.IO, OperationTag.INITIALIZATION],
    parameter_schema=[
        ParameterDef(
            name="metadata_group_columns_to_keep",
            type="list_or_str",
            required=True,
            default=None,
            label="Metadata group columns",
            help="Metadata columns to keep from batch info, or 'all'.",
            example="e.g.: Diagnosis, Tumour Type",
        ),
        ParameterDef(
            name="data_prefix",
            type="str",
            required=False,
            default="Area:",
            label="Intensity column prefix",
            help="Prefix used to identify intensity columns.",
            example="e.g.: Area:",
        ),
        ParameterDef(
            name="cpdID_from_column",
            type="bool",
            required=False,
            default=False,
            label="Use existing column as cpdID",
            help="If False, cpdID is created from m/z and RT. If True, one existing column is used.",
            example="e.g.: False",
        ),
        ParameterDef(
            name="cpdID_columns",
            type="list",
            required=False,
            default=["m/z", "RT [min]"],
            label="cpdID columns",
            help="Use two columns for m/z+RT mode, or one column if cpdID_from_column=True.",
            example="e.g.: m/z, RT [min]",
        ),
        ParameterDef(
            name="more_batches",
            type="bool",
            required=False,
            default=False,
            label="Multiple batches",
            help="If False, all samples are treated as one batch.",
            example="e.g.: False",
        ),
        ParameterDef(
            name="datetime_format",
            type="str",
            required=False,
            default="%d/%m/%Y %H:%M:%S",
            label="Datetime format",
            help="Format used to parse Creation Date in batch info.",
            example="e.g.: %d/%m/%Y %H:%M:%S",
        ),
        ParameterDef(
            name="qc_samples_distinguisher",
            type="str",
            required=False,
            default="Quality Control",
            label="Quality Control Samples Name",
            help="Name of the Quality Control samples (in Sample Type column)",
            example="e.g.: Quality Control, QC, ...",
        ),
        ParameterDef(
            name="blank_samples_distinguisher",
            type="str",
            required=False,
            default="Blank",
            label="Blank Samples Name",
            help="Name of the Blank samples (in Sample Type column)",
            example="e.g.: Blank, Blanks, Blank Samples, ...",
        ),
        ParameterDef(
            name="standard_samples_distinguisher",
            type="str",
            required=False,
            default="Standard",
            label="Standard Samples Name",
            help="Name of the Standard samples (in Sample Type column)",
            example="e.g.: Standards, ...",
        ),
        ParameterDef(
            name="dil_distinguisher",
            type="str",
            required=False,
            default="dilQC",
            label="dilution series distinguisher Name",
            help="How the dilution QCs are distinguished in the file names",
            example="e.g.: dilQC, dilutionQC, ...",
        ),
        ParameterDef(
            name="conc_distinguisher",
            type="str",
            required=False,
            default="dilQC_",
            label="dilution series concentration prefix",
            help="The prefix to the concentration values in the name.",
            example="e.g.: dilQC_ if you have dilQC_50, etc...",
        ),
    ],
    requires=["files"],
    produces=[
        "data",
        "variable_metadata",
        "batch_info",
        "metadata",
        "batch",
        "QC_samples",
        "blank_samples",
        "dilution_series_samples",
        "dil_concentrations",
        "standard_samples",
        "report",
        "main_folder",
    ],
)
def initializer_compound_discoverer(
    state: WorkflowState,
    metadata_group_columns_to_keep,
    data_prefix="Area:",
    cpdID_from_column=False,
    cpdID_columns=None,
    more_batches=False,
    datetime_format="%d/%m/%Y %H:%M:%S",
    qc_samples_distinguisher="Quality Control",
    blank_samples_distinguisher="Blank",
    standard_samples_distinguisher="Standard",
    dil_distinguisher="dilQC",
    conc_distinguisher="dilQC_",
):
    if cpdID_columns is None:
        cpdID_columns = ["m/z", "RT [min]"]

    if not hasattr(state, "files") or state.files is None:
        raise ValueError("state.files is missing. Expected {'data': path, 'batch_info': path}.")

    data_input_file_name = state.files.get("data")
    batch_info_input_file_name = state.files.get("batch_info")

    if data_input_file_name is None:
        raise ValueError("No data file found in state.files['data'].")

    if batch_info_input_file_name is None:
        raise ValueError("No batch info file found in state.files['batch_info'].")

    # Initialize folders first, then report.
    _initializer_folders(state)
    _initializer_report(state)

    # Load raw Compound Discoverer table.
    _, data_load_info = _loader_data(
        state,
        data_input_file_name=data_input_file_name,
    )

    # Create cpdID.
    if cpdID_from_column:
        if isinstance(cpdID_columns, str):
            cpdID_col = cpdID_columns
        else:
            cpdID_col = cpdID_columns[0]

        _add_cpdID_from_column(
            state,
            cpdID_col=cpdID_col,
        )

    else:
        if isinstance(cpdID_columns, str) or len(cpdID_columns) < 2:
            raise ValueError(
                "When cpdID_from_column=False, cpdID_columns must contain two columns: "
                "[mz_col, rt_col]."
            )

        _add_cpdID(
            state,
            mz_col=cpdID_columns[0],
            rt_col=cpdID_columns[1],
        )

    # Extract variable metadata before reducing data to intensity matrix.
    _extracter_variable_metadata(
        state,
        column_index_ranges=[[10, 15], [18, 23]],
    )

    # Extract intensity matrix.
    _extracter_data(
        state,
        prefix=data_prefix,
    )

    # Load batch info.
    _, batch_info_load_info = _loader_batch_info(
        state,
        batch_info_input_file_name=batch_info_input_file_name,
    )

    # Reorder samples.
    if more_batches:
        raise NotImplementedError(
            "Multiple-batch initialization is not implemented yet. "
            "Use more_batches=False for now."
        )

    _batch_by_name_reorder(
        state,
        distinguisher=None,
        datetime_format=datetime_format,
    )

    # Extract metadata.
    _extracter_metadata(
        state,
        group_columns_to_keep=metadata_group_columns_to_keep,
    )

    # Initialize sample lists for later filters/corrections.
    _initialize_sample_type_lists(
    state,
    qc_samples_distinguisher=qc_samples_distinguisher,
    blank_samples_distinguisher=blank_samples_distinguisher,
    standard_samples_distinguisher=standard_samples_distinguisher,
    dil_distinguisher=dil_distinguisher,
    conc_distinguisher=conc_distinguisher,
)

    if state.report is not None:
        state.report.add_together(
            [
                ("text", "Compound Discoverer dataset initialization completed.", "bold"),
                ("text", f"Number of features: {state.data.shape[0]}"),
                ("text", f"Number of samples: {state.data.shape[1] - 1}"),
                ("text", f"Data load info: {data_load_info}", "italic"),
                ("text", f"Batch info load info: {batch_info_load_info}", "italic"),
                "line",
            ]
        )

    return {
        "initialized": True,
        "format": "compound_discoverer",
        "n_features": int(state.data.shape[0]),
        "n_samples": int(state.data.shape[1] - 1),
        "data_load_info": data_load_info,
        "batch_info_load_info": batch_info_load_info,
        "metadata_columns": list(state.metadata.columns)
        if state.metadata is not None
        else [],
        "variable_metadata_columns": list(state.variable_metadata.columns)
        if state.variable_metadata is not None
        else [],
        "n_qc_samples": len(state.QC_samples) if state.QC_samples is not None else 0,
        "n_blank_samples": len(state.blank_samples) if state.blank_samples is not None else 0,
        "n_dilution_series_samples": len(state.dilution_series_samples)
        if state.dilution_series_samples is not None
        else 0,
        "n_standard_samples": len(state.standard_samples)
        if state.standard_samples is not None
        else 0,
    }


# helping functions copied from previous version of the module

def _initializer_report(state: WorkflowState):
    """
    Initialize the report object.
    """
    main_folder = state.main_folder

    if main_folder is None:
        raise ValueError("state.main_folder is not set. Initialize folders first.")

    report_file_name = getattr(state, "report_file_name", None)

    if report_file_name is None:
        report_file_name = "pyspresso_report"

    if not report_file_name.endswith(".pdf"):
        report_file_name = report_file_name + ".pdf"

    report_path = os.path.join(main_folder, report_file_name)
    title_text = main_folder

    report = Report(name=report_path, title=title_text)
    report.initialize_report()

    logo_path = "pyspresso_logo.png"

    if os.path.exists(logo_path):
        report.add_image(logo_path, max_width=420, max_height=260)
    else:
        print("Logo not found, skipping logo addition to the report.")

    processed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pyspresso_version = getattr(state, "pyspresso_version", None)

    report.add_text(
        f"Processed at: {processed_time}",
        style="italic",
        alignment="center",
        font_size=10,
    )

    if pyspresso_version is not None:
        report.add_text(
            f"PySPRESSO version: {pyspresso_version}",
            style="italic",
            alignment="center",
            font_size=10,
        )

    state.report = report
    state.report_path = report_path
    state.report_file_name = report_file_name

    print("Report initialized.")

    return state.report
    
def _initializer_folders(state: WorkflowState):
    """
    Initialize output folders.
    """
    main_folder = getattr(state, "main_folder", None)

    if main_folder is None:
        workflow_id = getattr(state, "workflow_id", "workflow")
        main_folder = os.path.join("outputs", str(workflow_id))
        state.main_folder = main_folder

    folders = [
        main_folder,
        os.path.join(main_folder, "figures"),
        os.path.join(main_folder, "statistics"),
        os.path.join(main_folder, "dropped_features"),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Folders initialized.")

    return main_folder

def _add_cpdID(
    state: WorkflowState,
    mz_col="m/z",
    rt_col="RT [min]",
    round_mz_col=5,
    round_rt_col=3,
):
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if mz_col not in data.columns:
        raise ValueError(f"Column not found: {mz_col}")

    if rt_col not in data.columns:
        raise ValueError(f"Column not found: {rt_col}")

    mz = pd.to_numeric(data[mz_col], errors="coerce")
    rt = pd.to_numeric(data[rt_col], errors="coerce")

    if mz.isna().any():
        raise ValueError(f"Column {mz_col} contains non-numeric values.")

    if rt.isna().any():
        raise ValueError(f"Column {rt_col} contains non-numeric values.")

    data["cpdID"] = (
        "M"
        + mz.round(round_mz_col).astype(float).astype(str)
        + "-T"
        + rt.round(round_rt_col).astype(float).astype(str)
    )

    duplicate_count = data.groupby("cpdID").cumcount()
    is_duplicate = duplicate_count > 0

    data.loc[is_duplicate, "cpdID"] = (
        data.loc[is_duplicate, "cpdID"]
        + "_"
        + duplicate_count.loc[is_duplicate].astype(str)
    )

    state.data = data

    print("Compound ID was added to the data.")

    if report is not None:
        text = (
            "cpdID column was added to the data calculated from m/z and RT. "
            'Matching IDs were distinguished by adding "_1", "_2", etc.'
        )
        report.add_together([("text", text), "line"])

    return state.data
    
def _add_cpdID_from_column(
    state: WorkflowState,
    cpdID_col,
):
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if cpdID_col not in data.columns:
        raise ValueError(f"Column not found: {cpdID_col}")

    data["cpdID"] = data[cpdID_col].astype(str).str.replace(";", ",,", regex=False)

    duplicate_count = data.groupby("cpdID").cumcount()
    is_duplicate = duplicate_count > 0

    data.loc[is_duplicate, "cpdID"] = (
        data.loc[is_duplicate, "cpdID"]
        + "_"
        + duplicate_count.loc[is_duplicate].astype(str)
    )

    state.data = data

    print("Compound ID was added to the data from column: " + str(cpdID_col))

    if report is not None:
        text = "cpdID column was added from existing column: " + str(cpdID_col) + "."
        report.add_together([("text", text), "line"])

    return state.data

def _extracter_variable_metadata(
    state: WorkflowState,
    columns_by_name=None,
    column_index_ranges=None,
):
    """
    Extract variable metadata from state.data.
    """
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if columns_by_name is None:
        columns_by_name = ["cpdID", "Name", "Formula"]

    if column_index_ranges is None:
        column_index_ranges = [[10, 15], [18, 23]]

    missing_columns = [
        col for col in columns_by_name
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "These variable metadata columns were not found in data: "
            + str(missing_columns)
        )

    variable_metadata = data[columns_by_name].copy()

    for column_index_range in column_index_ranges:
        if isinstance(column_index_range, (list, tuple)):
            if len(column_index_range) != 2:
                raise ValueError(
                    "Column index range must contain exactly two values: "
                    + str(column_index_range)
                )

            start = int(column_index_range[0])
            end = int(column_index_range[1])

            if start < 0 or end > data.shape[1] or start >= end:
                raise ValueError(
                    "Column index range "
                    + str(column_index_range)
                    + " is out of bounds."
                )

            cols_to_add = [
                col for col in data.columns[start:end]
                if col not in variable_metadata.columns
            ]

            if cols_to_add:
                variable_metadata = variable_metadata.join(data[cols_to_add])

        else:
            column_index = int(column_index_range)

            if column_index < 0 or column_index >= data.shape[1]:
                raise ValueError(
                    "Column index "
                    + str(column_index)
                    + " is out of bounds."
                )

            col = data.columns[column_index]

            if col not in variable_metadata.columns:
                variable_metadata = variable_metadata.join(data.iloc[:, column_index])

    state.variable_metadata = variable_metadata.reset_index(drop=True)

    print("Variable metadata was extracted from the data.")

    if report is not None:
        report.add_together(
            [
                ("text", "variable-metadata matrix was created."),
                "line",
            ]
        )

    return state.variable_metadata

def _extracter_data(
    state: WorkflowState,
    prefix="Area:",
):
    """
    Extract intensity matrix from the full loaded data.
    """
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if "cpdID" not in data.columns:
        raise ValueError("No cpdID column found in state.data.")

    temp_data = data.copy()

    area_columns = [
        col for col in temp_data.columns
        if str(col).startswith(prefix)
    ]

    if len(area_columns) == 0:
        raise ValueError(
            "No intensity columns found with prefix: " + str(prefix)
        )

    extracted_data = temp_data[["cpdID"]].join(temp_data[area_columns])
    extracted_data.iloc[:, 1:] = extracted_data.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce",
    )

    extracted_data.fillna(0, inplace=True)

    state.data = extracted_data

    print("Important columns were kept in the data and rest filtered out.")

    if report is not None:
        report.add_together(
            [
                ("text", "data matrix was created."),
                "line",
            ]
        )

    return state.data

def _batch_by_name_reorder(
    state: WorkflowState,
    distinguisher="Batch",
    distinguisher_col="File Name",
    datetime_col="Creation Date",
    datetime_format="%d/%m/%Y %H:%M:%S",
    sample_id_col="Study File ID",
    sample_type_col="Sample Type",
):
    """
    Reorder data based on batch_info and creation date.
    Also creates batch_info['Batch'] and state.batch.
    """
    data = state.data
    report = state.report
    batch_info = state.batch_info

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if batch_info is None:
        raise ValueError("No batch info loaded in state.batch_info.")

    if "cpdID" not in data.columns:
        raise ValueError("Expected cpdID column in state.data.")

    if sample_id_col not in batch_info.columns:
        raise ValueError(f"{sample_id_col} not found in batch_info.")

    if sample_type_col not in batch_info.columns:
        raise ValueError(f"{sample_type_col} not found in batch_info.")

    batch_info = batch_info.copy()
    names = data.columns[1:].to_list()

    if datetime_col is None:
        print(
            "Assuming batch_info has samples in the correct order already, "
            "since no datetime_col is provided."
        )
    else:
        if datetime_col not in batch_info.columns:
            raise ValueError(f"{datetime_col} not found in batch_info.")

        if datetime_format == "order":
            batch_info[datetime_col] = pd.to_numeric(
                batch_info[datetime_col],
                errors="coerce",
            )
        else:
            batch_info[datetime_col] = pd.to_datetime(
                batch_info[datetime_col],
                format=datetime_format,
                errors="raise",
            )

        batch_info = batch_info.sort_values(datetime_col)
        batch_info = batch_info.reset_index(drop=True)

    if distinguisher is None:
        batch_info["Batch"] = [
            "all_one_batch"
            for _ in range(len(batch_info.index))
        ]

        if report is not None:
            report.add_together(
                [
                    ("text", "All samples are in one batch."),
                    "line",
                ]
            )

    else:
        if distinguisher_col not in batch_info.columns:
            raise ValueError(f"{distinguisher_col} not found in batch_info.")

        if distinguisher == "":
            batch_info["Batch"] = batch_info[distinguisher_col].astype(str)
            print("Batch names taken directly from column: " + distinguisher_col)

        else:
            batches_column = []

            for file_name in batch_info[distinguisher_col].tolist():
                split_name = re.split(r"[_|\\]", str(file_name))

                found_batch = None

                for i, part in enumerate(split_name):
                    if part == distinguisher and i + 1 < len(split_name):
                        found_batch = split_name[i + 1]
                        break

                if found_batch is None:
                    found_batch = "unknown_batch"

                batches_column.append(found_batch)

            if len(batches_column) == 0:
                raise ValueError(
                    "No matches found for distinguisher: "
                    + str(distinguisher)
                )

            batch_info["Batch"] = batches_column

    not_found = []
    not_found_indexes = []
    new_data_order = []
    remaining_names = names.copy()

    for i, sample_id in enumerate(batch_info[sample_id_col].tolist()):
        found = False
        sample_id_str = str(sample_id)

        for name in remaining_names:
            name_str = str(name)

            if re.search(rf"\({re.escape(sample_id_str)}\)", name_str):
                found = True
                new_data_order.append(name)
                remaining_names.remove(name)
                break

            if name_str == sample_id_str:
                found = True
                new_data_order.append(name)
                remaining_names.remove(name)
                break

        if not found:
            not_found_indexes.append(i)
            not_found.append([sample_id, batch_info[sample_type_col].tolist()[i]])

    if len(new_data_order) == 0:
        raise ValueError(
            "No sample columns could be matched between data and batch_info. "
            "Check Study File ID and data column names."
        )

    print("New data order based on batch info:")
    print(new_data_order)
    print("Data reordered based on creation date from batch info.")
    print("Not found: " + str(len(not_found)) + " ; being: " + str(not_found))
    print(
        "Names not identified: "
        + str(len(remaining_names))
        + " ; being: "
        + str(remaining_names)
    )

    data = data[["cpdID"] + new_data_order].copy()

    batch_info = batch_info.drop(batch_info.index[not_found_indexes])
    batch_info = batch_info.reset_index(drop=True)

    state.batch_info = batch_info
    state.data = data
    state.batch = batch_info["Batch"].tolist()

    if report is not None:
        text0 = "Batch information was used to reorder samples."

        if distinguisher is None:
            text1 = 'All samples are in one batch. Thus "all_one_batch" was used.'
        else:
            text1 = (
                "Batches were distinguished in the File Name using: "
                + str(distinguisher)
            )

        text2 = "Not found: " + str(len(not_found)) + " ; being: " + str(not_found)
        text3 = (
            "Names not identified: "
            + str(len(remaining_names))
            + " ; being: "
            + str(remaining_names)
        )

        report.add_together(
            [
                ("text", text0),
                ("text", text1, "italic"),
                ("text", text2, "italic"),
                ("text", text3, "italic"),
                ("table", batch_info),
                "line",
            ]
        )

    return state.data, state.batch_info

def _extracter_metadata(
    state: WorkflowState,
    group_columns_to_keep,
    always_keep_columns=None,
):
    """
    Extract metadata from batch_info.
    """
    data = state.data
    report = state.report
    batch_info = state.batch_info

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if batch_info is None:
        raise ValueError("No batch_info loaded in state.batch_info.")

    if always_keep_columns is None:
        always_keep_columns = [
            "Study File ID",
            "File Name",
            "Creation Date",
            "Sample Type",
            "Polarity",
            "Batch",
        ]

    if isinstance(group_columns_to_keep, str):
        if group_columns_to_keep.strip().lower() == "all":
            columns_to_keep = batch_info.columns.tolist()
        else:
            columns_to_keep = [
                col.strip()
                for col in group_columns_to_keep.split(",")
                if col.strip()
            ]
            columns_to_keep = always_keep_columns + columns_to_keep

    else:
        columns_to_keep = always_keep_columns + list(group_columns_to_keep)

    columns_to_keep = list(dict.fromkeys(columns_to_keep))

    missing_columns = [
        col for col in columns_to_keep
        if col not in batch_info.columns
    ]

    if missing_columns:
        raise ValueError(
            "These metadata columns were not found in batch_info: "
            + str(missing_columns)
        )

    sample_columns = data.drop(columns=["cpdID"]).columns.tolist()

    if len(sample_columns) != len(batch_info):
        raise ValueError(
            "Number of sample columns in data does not match rows in batch_info "
            "after reordering."
        )

    metadata = batch_info[columns_to_keep].copy()
    metadata["Sample File"] = sample_columns

    state.metadata = metadata

    if report is not None:
        text = (
            "metadata matrix was created from batch_info by choosing columns: "
            + str(columns_to_keep)
            + "."
        )
        report.add_together([("text", text), "line"])

    return state.metadata

def _clean_loaded_table(df):
    """
    Basic cleanup after loading a table.
    """
    df = df.copy()

    # Drop fully empty rows and columns.
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Remove unnamed all-empty columns if they sneak in.
    df.columns = [str(col).strip() for col in df.columns]

    return df.reset_index(drop=True)

def _looks_like_valid_table(df, min_columns=2, min_rows=1):
    """
    Very simple sanity check to avoid accepting wrongly parsed CSV files.
    """
    if df is None:
        return False

    if df.shape[0] < min_rows:
        return False

    if df.shape[1] < min_columns:
        return False

    return True

def _read_excel_auto(path, min_columns=2):
    """
    Read an Excel file.

    If there are multiple sheets, choose the first sheet that looks like a real table.
    """
    try:
        excel_file = pd.ExcelFile(path)
    except Exception as exc:
        raise ValueError(f"Could not open Excel file: {path}. Error: {exc}")

    errors = []

    for sheet_name in excel_file.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            df = _clean_loaded_table(df)

            if _looks_like_valid_table(df, min_columns=min_columns):
                info = {
                    "file_type": "excel",
                    "sheet_name": sheet_name,
                    "separator": None,
                    "encoding": None,
                }

                return df, info

            errors.append(
                f"Sheet {sheet_name!r} did not look like a valid table: shape={df.shape}"
            )

        except Exception as exc:
            errors.append(f"Sheet {sheet_name!r} failed: {exc}")

    raise ValueError(
        "Could not find a usable sheet in Excel file: "
        + str(path)
        + ". Attempts: "
        + str(errors)
    )

def _read_text_table_auto(path, min_columns=2):
    """
    Read CSV/TXT/TSV-like files by trying common encodings and separators.

    First tries pandas automatic separator inference.
    Then falls back to common separators.
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "ISO-8859-1",
        "cp1250",
        "latin1",
    ]

    separators = [
        None,   # pandas tries to infer separator with engine="python"
        ";",
        ",",
        "\t",
        "|",
    ]

    errors = []

    for encoding in encodings:
        for separator in separators:
            try:
                if separator is None:
                    df = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                    )
                else:
                    df = pd.read_csv(
                        path,
                        sep=separator,
                        encoding=encoding,
                    )

                df = _clean_loaded_table(df)

                if _looks_like_valid_table(df, min_columns=min_columns):
                    info = {
                        "file_type": "text",
                        "sheet_name": None,
                        "separator": "auto" if separator is None else separator,
                        "encoding": encoding,
                    }

                    return df, info

                errors.append(
                    f"encoding={encoding}, separator={separator!r} produced shape={df.shape}"
                )

            except Exception as exc:
                errors.append(
                    f"encoding={encoding}, separator={separator!r} failed: {exc}"
                )

    raise ValueError(
        "Could not load text table: "
        + str(path)
        + ". Tried common encodings and separators. First errors: "
        + str(errors[:10])
    )

def _load_table_auto(file_path, min_columns=2):
    """
    Load CSV/TXT/TSV/XLSX/XLSM/XLS table automatically.

    Returns
    -------
    df : pandas.DataFrame
    info : dict
        Information about detected file type, sheet, separator, and encoding.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    excel_suffixes = {
        ".xlsx",
        ".xlsm",
        ".xltx",
        ".xltm",
        ".xls",
    }

    text_suffixes = {
        ".csv",
        ".txt",
        ".tsv",
    }

    if suffix in excel_suffixes:
        return _read_excel_auto(path, min_columns=min_columns)

    if suffix in text_suffixes:
        return _read_text_table_auto(path, min_columns=min_columns)

    # Unknown extension: try text first, then Excel.
    try:
        return _read_text_table_auto(path, min_columns=min_columns)
    except Exception:
        return _read_excel_auto(path, min_columns=min_columns)
    
def _loader_data(state: WorkflowState, data_input_file_name):
    report = state.report

    data, load_info = _load_table_auto(
        data_input_file_name,
        min_columns=2,
    )

    state.data = data

    print("Data loaded.")
    print("Load info:", load_info)

    if report is not None:
        text = (
            "Data were loaded from: "
            + str(data_input_file_name)
            + " (Compound Discoverer data). "
            + "Detected loading settings: "
            + str(load_info)
            + "."
        )

        report.add_together(
            [
                ("text", text),
                ("table", state.data),
                "line",
            ]
        )

    return state.data, load_info

def _loader_batch_info(state: WorkflowState, batch_info_input_file_name):
    report = state.report

    batch_info, load_info = _load_table_auto(
        batch_info_input_file_name,
        min_columns=2,
    )

    state.batch_info = batch_info

    print("Batch info loaded.")
    print("Load info:", load_info)

    if report is not None:
        text = (
            "Batch info matrix was loaded from: "
            + str(batch_info_input_file_name)
            + ". Detected loading settings: "
            + str(load_info)
            + "."
        )

        report.add_together(
            [
                ("text", text),
                "line",
            ]
        )

    return state.batch_info, load_info

def _initialize_sample_type_lists(
    state: WorkflowState,
    qc_samples_distinguisher="Quality Control",
    blank_samples_distinguisher="Blank",
    standard_samples_distinguisher="Standard",
    dil_distinguisher="dilQC",
    conc_distinguisher="dilQC_",
):
    """
    Initialize sample lists used by filters and corrections.

    QC, blank, and standard samples are identified from metadata['Sample Type'].
    Dilution-series samples are identified from metadata['Sample File'] names.
    """
    metadata = state.metadata

    if metadata is None:
        raise ValueError("No metadata found in state.metadata.")

    if "Sample Type" not in metadata.columns:
        raise ValueError("metadata must contain 'Sample Type' column.")

    if "Sample File" not in metadata.columns:
        raise ValueError("metadata must contain 'Sample File' column.")

    sample_type = metadata["Sample Type"].astype(str)
    sample_file = metadata["Sample File"].astype(str)

    # QC samples
    state.QC_samples = metadata.loc[
        sample_type == qc_samples_distinguisher,
        "Sample File",
    ].tolist()

    # Blank samples
    state.blank_samples = metadata.loc[
        sample_type == blank_samples_distinguisher,
        "Sample File",
    ].tolist()

    # Standard samples
    state.standard_samples = metadata.loc[
        sample_type == standard_samples_distinguisher,
        "Sample File",
    ].tolist()

    # Dilution-series samples by name
    if dil_distinguisher is None or dil_distinguisher == "":
        state.dilution_series_samples = []
        state.dil_concentrations = []
    else:
        dilution_mask = sample_file.str.contains(
            str(dil_distinguisher),
            case=False,
            na=False,
            regex=False,
        )

        state.dilution_series_samples = metadata.loc[
            dilution_mask,
            "Sample File",
        ].tolist()

        # Optional: try to extract concentration/order after conc_distinguisher.
        # Example: "sample_dilQC_0.25" with conc_distinguisher="dilQC_"
        dil_concentrations = []

        for name in state.dilution_series_samples:
            match = re.search(
                re.escape(str(conc_distinguisher)) + r"([0-9]+(?:[.,][0-9]+)?)",
                str(name),
            )

            if match:
                value = match.group(1).replace(",", ".")

                try:
                    dil_concentrations.append(float(value))
                except ValueError:
                    dil_concentrations.append(None)
            else:
                dil_concentrations.append(None)

        state.dil_concentrations = dil_concentrations

    print("Sample type lists initialized.")
    print(f"QC samples: {len(state.QC_samples)}")
    print(f"Blank samples: {len(state.blank_samples)}")
    print(f"Dilution-series samples: {len(state.dilution_series_samples)}")
    print(f"Standard samples: {len(state.standard_samples)}")

    if state.report is not None:
        state.report.add_together(
            [
                ("text", "Sample type lists were initialized.", "bold"),
                ("text", f"QC samples distinguished by Sample Type == {qc_samples_distinguisher!r}: {len(state.QC_samples)}"),
                ("text", f"Blank samples distinguished by Sample Type == {blank_samples_distinguisher!r}: {len(state.blank_samples)}"),
                ("text", f"Standard samples distinguished by Sample Type == {standard_samples_distinguisher!r}: {len(state.standard_samples)}"),
                ("text", f"Dilution-series samples distinguished by sample name containing {dil_distinguisher!r}: {len(state.dilution_series_samples)}"),
                "line",
            ]
        )

    return {
        "QC_samples": state.QC_samples,
        "blank_samples": state.blank_samples,
        "dilution_series_samples": state.dilution_series_samples,
        "dil_concentrations": state.dil_concentrations,
        "standard_samples": state.standard_samples,
    }