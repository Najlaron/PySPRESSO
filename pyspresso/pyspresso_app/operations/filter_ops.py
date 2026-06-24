import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def _write_versioned_txt(folder, base_name, lines, digits=3):
    """
    Helper function to write a versioned text file.

    This is used by filter operations to save lists of removed features,
    especially when the list is too long for the report.
    """
    os.makedirs(folder, exist_ok=True)

    pat = re.compile(rf"^{re.escape(base_name)}_v(\d{{{digits}}})\.txt$")
    max_v = 0

    for fn in os.listdir(folder):
        m = pat.match(fn)
        if m:
            max_v = max(max_v, int(m.group(1)))

    path = os.path.join(folder, f"{base_name}_v{max_v + 1:0{digits}d}.txt")

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")

    return path


def _filter_match_variable_metadata(data, variable_metadata):
    """
    Update variable_metadata based on the filtered data.

    Keeps only rows from variable_metadata whose cpdID still exists in data.
    """
    if variable_metadata is None:
        return None

    if "cpdID" not in data.columns:
        return variable_metadata.reset_index(drop=True)

    if "cpdID" not in variable_metadata.columns:
        return variable_metadata.reset_index(drop=True)

    variable_metadata = variable_metadata[
        variable_metadata["cpdID"].isin(data["cpdID"])
    ].copy()

    variable_metadata.reset_index(drop=True, inplace=True)

    return variable_metadata


def _get_dropped_features_folder(state: WorkflowState):
    """
    Get the folder where dropped-feature lists should be saved.

    Old PySPRESSO used:
        self.main_folder + '/dropped_features'

    Here we use:
        state.main_folder + '/dropped_features'

    If state.main_folder does not exist yet, we create a fallback folder.
    """
    main_folder = getattr(state, "main_folder", None)

    if main_folder is None:
        workflow_id = getattr(state, "workflow_id", "workflow")
        main_folder = os.path.join("outputs", str(workflow_id))
        state.main_folder = main_folder

    dropped_features_folder = os.path.join(main_folder, "dropped_features")
    return dropped_features_folder


def _add_artifact_if_available(state: WorkflowState, path, artifact_type, description):
    """
    Optional helper for future frontend use.

    If WorkflowState has an artifacts list, the generated text file is stored there.
    If not, nothing happens.
    """
    if not hasattr(state, "artifacts"):
        return

    state.artifacts.append(
        {
            "type": artifact_type,
            "path": path,
            "description": description,
        }
    )


# ------------------------------------------------------------
# Filter operations
# ------------------------------------------------------------


@register_operation(
    id="filter_missing_values",
    label="Filter Missing Values",
    description=(
        "Filter out features with too many missing values or zero values. "
        "Filtering is first applied within QC samples, then across all samples."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="qc_threshold",
            type="float",
            required=False,
            default=0.8,
            label="QC threshold",
            help=(
                "Required fraction of present values within QC samples. "
                "For example, 0.8 means at least 80% presence in QC samples."
            ),
        ),
        ParameterDef(
            name="sample_threshold",
            type="float",
            required=False,
            default=0.5,
            label="Sample threshold",
            help=(
                "Required fraction of present values across all samples. "
                "For example, 0.5 means at least 50% presence across all samples."
            ),
        ),
    ],
    requires=["data", "QC_samples"],
    produces=["data", "variable_metadata"],
)
def filter_missing_values(
    state: WorkflowState,
    qc_threshold: float = 0.8,
    sample_threshold: float = 0.5,
):
    """
    Filter out features with high number of missing values.

    Missing means:
        - NaN
        - 0

    Filtering order:
        1. Within QC samples
        2. Across all samples
    """
    data = state.data
    report = state.report
    QC_samples = state.QC_samples

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if QC_samples is None:
        raise ValueError("No QC samples defined in state.QC_samples.")

    if not 0 <= qc_threshold <= 1:
        raise ValueError("qc_threshold must be between 0 and 1.")

    if not 0 <= sample_threshold <= 1:
        raise ValueError("sample_threshold must be between 0 and 1.")

    # A) WITHIN QC SAMPLES ---------------------------------------------

    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]

    qc_cols = data.columns[1:][is_qc_sample]
    qc_count = len(qc_cols)

    if qc_count == 0:
        raise ValueError(
            "No QC sample columns found in data. "
            "Check whether state.QC_samples contains names matching data columns."
        )

    qc_block = data[qc_cols]

    # Presence = not NaN and not zero.
    qc_present = (~qc_block.isna()) & (qc_block != 0)
    presence_frac_QC = qc_present.sum(axis=1) / qc_count

    # Compute removals before filtering data.
    removed_mask = presence_frac_QC < qc_threshold
    removed_idx = data.index[removed_mask]
    removed_count = int(removed_mask.sum())

    removed_ids = (
        data.loc[removed_idx, "cpdID"].tolist()
        if "cpdID" in data.columns
        else removed_idx.tolist()
    )

    details = f" ; being: {removed_ids}" if removed_ids else ""

    print(
        f"Number of features removed for QC threshold ({qc_threshold * 100:.0f}%): "
        f"within QC samples: {removed_count}{details}"
    )

    # Filter data.
    data = data.loc[~removed_mask, :].reset_index(drop=True)

    # B) ACROSS ALL SAMPLES ---------------------------------------------

    all_cols = data.columns[1:]
    all_count = len(all_cols)

    if all_count == 0:
        raise ValueError("No sample columns found after the first column.")

    all_block = data[all_cols]

    # Presence = not NaN and not zero.
    all_present = (~all_block.isna()) & (all_block != 0)
    presence_frac_ALL = all_present.sum(axis=1) / all_count

    # Compute removals before filtering data.
    removed_mask_all = presence_frac_ALL < sample_threshold
    removed_idx_all = data.index[removed_mask_all]
    removed_count_all = int(removed_mask_all.sum())

    removed_ids_all = (
        data.loc[removed_idx_all, "cpdID"].tolist()
        if "cpdID" in data.columns
        else removed_idx_all.tolist()
    )

    details_all = f" ; being: {removed_ids_all}" if removed_ids_all else ""

    print(
        f"Number of features removed for sample threshold ({sample_threshold * 100:.0f}%): "
        f"within all samples: {removed_count_all}{details_all}"
    )

    # Filter data.
    data = data.loc[~removed_mask_all, :].reset_index(drop=True)

    # REPORTING ---------------------------------------------------------

    dropped_features_folder = _get_dropped_features_folder(state)

    txt_path = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name=(
            "removed_features_missing_values_qc_threshold_"
            + str(int(qc_threshold * 100))
            + "pct"
        ),
        lines=removed_ids,
    )

    txt_path_all = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name=(
            "removed_features_missing_values_sample_threshold_"
            + str(int(sample_threshold * 100))
            + "pct"
        ),
        lines=removed_ids_all,
    )

    _add_artifact_if_available(
        state,
        path=txt_path,
        artifact_type="dropped_features",
        description="Features removed by missing-values QC threshold.",
    )

    _add_artifact_if_available(
        state,
        path=txt_path_all,
        artifact_type="dropped_features",
        description="Features removed by missing-values sample threshold.",
    )

    text0 = (
        "Features with missing values over the threshold ("
        + str(qc_threshold * 100)
        + "%) within QC samples were removed. Number of features removed: "
        + str(removed_count)
    )

    if removed_count > 0 and removed_count < 25:
        text1 = " ;being: " + str(removed_ids)
    elif removed_count >= 25:
        text1 = "The list of removed features is long and saved in: " + txt_path
    else:
        text1 = ""

    text2 = (
        "Features with missing values over the threshold ("
        + str(sample_threshold * 100)
        + "%) across all samples were removed. Number of features removed: "
        + str(removed_count_all)
    )

    if removed_count_all > 0 and removed_count_all < 25:
        text3 = " ;being: " + str(removed_ids_all)
    elif removed_count_all >= 25:
        text3 = "The list of removed features is long and saved in: " + txt_path_all
    else:
        text3 = ""

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                ("text", text2),
                ("text", text3),
                "line",
            ]
        )
        report.add_pagebreak()

    # Update state ------------------------------------------------------

    state.data = data
    state.data.reset_index(drop=True, inplace=True)

    state.variable_metadata = _filter_match_variable_metadata(
        data,
        state.variable_metadata,
    )

    if state.variable_metadata is not None:
        state.variable_metadata.reset_index(drop=True, inplace=True)

    return {
        "features_after": int(state.data.shape[0]),
        "removed_qc_count": removed_count,
        "removed_sample_count": removed_count_all,
        "removed_total_count": removed_count + removed_count_all,
        "qc_threshold": qc_threshold,
        "sample_threshold": sample_threshold,
        "removed_qc_file": txt_path,
        "removed_sample_file": txt_path_all,
    }


@register_operation(
    id="filter_blank_intensity_ratio",
    label="Filter Blank Intensity Ratio",
    description=(
        "Filter out features with QC sample intensity divided by blank intensity "
        "below the selected ratio."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="ratio",
            type="float",
            required=False,
            default=20,
            label="Sample/blank ratio",
            help="Minimum required ratio between QC sample intensity and blank intensity.",
        ),
        ParameterDef(
            name="setting",
            type="str",
            required=False,
            default="first",
            label="Blank setting",
            help=(
                "How to calculate blank intensity. "
                "Choose from: median, min, mean, first, last."
            ),
        ),
    ],
    requires=["data", "variable_metadata", "QC_samples", "blank_samples"],
    produces=["data", "variable_metadata"],
)
def filter_blank_intensity_ratio(
    state: WorkflowState,
    ratio: float = 20,
    setting: str = "first",
):
    """
    Filter out features with intensity sample/blank < ratio.

    The sample intensity is calculated as the median intensity of QC samples.
    The blank intensity is calculated according to setting:
        - median
        - min
        - mean
        - first
        - last
    """
    data = state.data
    variable_metadata = state.variable_metadata
    report = state.report
    QC_samples = state.QC_samples
    blank_samples = state.blank_samples

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if QC_samples is None:
        raise ValueError("No QC samples defined in state.QC_samples.")

    if ratio <= 0:
        raise ValueError("ratio must be greater than 0.")

    blank_threshold = ratio

    if blank_samples == []:
        print(
            "No blank samples defined in state.blank_samples. "
            "Skipping filter_blank_intensity_ratio step."
        )

        return {
            "skipped": True,
            "reason": "No blank samples defined. Blank sample list is empty.",
            "removed_count": 0,
            "ratio": ratio,
            "setting": setting,
        }

    if blank_samples is None or blank_samples is False:
        raise ValueError("No blank samples defined in state.blank_samples.")

    # Blank samples mask.
    is_blank_sample = [
        True if col in blank_samples else False for col in data.columns[1:]
    ]

    # QC samples mask.
    is_qc_sample = [True if col in QC_samples else False for col in data.columns[1:]]

    blank_cols = data.columns[1:][is_blank_sample]
    qc_cols = data.columns[1:][is_qc_sample]

    if len(blank_cols) == 0:
        raise ValueError(
            "No blank sample columns found in data. "
            "Check whether state.blank_samples contains names matching data columns."
        )

    if len(qc_cols) == 0:
        raise ValueError(
            "No QC sample columns found in data. "
            "Check whether state.QC_samples contains names matching data columns."
        )

    # Different approaches for blank intensity.
    if setting == "median":
        blank_intensities = data[blank_cols].median(axis=1)
    elif setting == "min":
        blank_intensities = data[blank_cols].min(axis=1)
    elif setting == "mean":
        blank_intensities = data[blank_cols].mean(axis=1)
    elif setting == "first":
        blank_intensities = data[blank_cols].iloc[:, 0]
    elif setting == "last":
        blank_intensities = data[blank_cols].iloc[:, -1]
    else:
        raise ValueError(
            "Setting not recognized. Choose from: "
            "'median', 'min', 'mean', 'first' or 'last'. "
            "(Chooses blank to use as reference value.)"
        )

    intensity_sample_blank = data[qc_cols].median(axis=1) / blank_intensities

    mask_removed = intensity_sample_blank < blank_threshold

    removed_ids = (
        data.loc[mask_removed, "cpdID"].tolist()
        if "cpdID" in data.columns
        else data.index[mask_removed].tolist()
    )

    removed_count = len(removed_ids)

    print(
        "Number of features removed for blank intensity ratio ("
        + str(ratio)
        + "): "
        + str(removed_count)
    )

    # Filter out features.
    data = data.loc[~mask_removed].reset_index(drop=True)

    # Update state.
    state.data = data
    state.variable_metadata = _filter_match_variable_metadata(
        data,
        variable_metadata,
    )

    # Save dropped-feature list.
    dropped_features_folder = _get_dropped_features_folder(state)

    txt_path = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name="removed_features_blank_intensity_ratio_" + str(int(ratio)),
        lines=removed_ids,
    )

    _add_artifact_if_available(
        state,
        path=txt_path,
        artifact_type="dropped_features",
        description="Features removed by blank intensity ratio filter.",
    )

    # REPORTING ---------------------------------------------------------

    text0 = (
        "Features with intensity sample/blank < "
        + str(blank_threshold)
        + " were removed. Number of features removed: "
        + str(removed_count)
    )

    if removed_count > 0 and removed_count < 25:
        text1 = " ;being: " + str(removed_ids)
    elif removed_count >= 25:
        text1 = "The list of removed features is long and saved in: " + txt_path
    else:
        text1 = ""

    if report is not None:
        elements = [("text", text0), ("text", text1)]
        elements.append("line")
        report.add_together(elements)

    return {
        "features_after": int(state.data.shape[0]),
        "removed_count": removed_count,
        "ratio": ratio,
        "setting": setting,
        "removed_features_file": txt_path,
        "skipped": False,
    }


@register_operation(
    id="filter_relative_standard_deviation",
    label="Filter Relative Standard Deviation",
    description=(
        "Filter out features whose QC relative standard deviation percentage "
        "is above the selected threshold."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="rsd_threshold",
            type="float",
            required=False,
            default=20,
            label="RSD threshold (%)",
            help="Maximum allowed QC RSD percentage.",
        ),
        ParameterDef(
            name="by_batch",
            type="bool",
            required=False,
            default=False,
            label="Calculate by batch",
            help="If True, calculate QC RSD per batch and use the median batch RSD.",
        ),
        ParameterDef(
            name="ignore_zero_qc",
            type="bool",
            required=False,
            default=False,
            label="Ignore zero QC values",
            help="If True, zero or negative QC values are treated as missing for RSD calculation.",
        ),
        ParameterDef(
            name="to_plot",
            type="bool_or_int",
            required=False,
            default=False,
            label="Plot removed features",
            help="False disables plotting, True plots 4 features, integer plots that many.",
        ),
        ParameterDef(
            name="min_qc_n",
            type="int",
            required=False,
            default=3,
            label="Minimum QC per batch",
            help="Minimum number of non-missing QC samples required in a batch.",
        ),
    ],
    requires=["data", "variable_metadata", "QC_samples"],
    produces=["data", "variable_metadata"],
)
def filter_relative_standard_deviation(
    state: WorkflowState,
    rsd_threshold: float = 20,
    by_batch: bool = False,
    ignore_zero_qc: bool = False,
    to_plot=False,
    min_qc_n: int = 3,
):
    data = state.data
    report = state.report
    QC_samples = state.QC_samples

    if data is None:
        raise ValueError("No data loaded in state.data.")

    variable_metadata = _filter_match_variable_metadata(data, state.variable_metadata)

    if QC_samples == []:
        print(
            "No QC samples defined in state.QC_samples. "
            "Skipping filter_relative_standard_deviation step."
        )
        return {
            "skipped": True,
            "reason": "No QC samples defined. QC sample list is empty.",
            "removed_count": 0,
        }

    if QC_samples is None or QC_samples is False:
        raise ValueError("No QC samples defined in state.QC_samples.")

    sample_cols = list(data.columns[1:])
    qc_cols = [c for c in sample_cols if c in QC_samples]

    if len(qc_cols) == 0:
        raise ValueError(
            "No QC sample columns found in data. "
            "Check names in state.QC_samples versus data columns."
        )

    qc_df = data[qc_cols].copy()

    # GLOBAL RSD --------------------------------------------------------

    if not by_batch:
        if ignore_zero_qc:
            qc_df = qc_df.mask(qc_df <= 0, np.nan)

        qc_mean = qc_df.mean(axis=1, skipna=True)
        qc_std = qc_df.std(axis=1, skipna=True)

        qc_rsd = (qc_std / qc_mean.replace(0, np.nan)) * 100

    # BATCH-WISE RSD ----------------------------------------------------

    else:
        batch_labels = getattr(state, "batch", None)

        if batch_labels is None:
            batch_info = getattr(state, "batch_info", None)

            if batch_info is not None and "Batch" in batch_info.columns:
                batch_labels = batch_info["Batch"].tolist()
            else:
                raise ValueError(
                    "Batch information not set. Run/set batch information first."
                )

        if len(batch_labels) != len(sample_cols):
            raise ValueError(
                "Length of state.batch does not match number of sample columns."
            )

        sample_to_batch = dict(zip(sample_cols, batch_labels))

        qc_cols_by_batch = {}
        for c in qc_cols:
            b = sample_to_batch.get(c, None)
            if b is None:
                continue
            qc_cols_by_batch.setdefault(b, []).append(c)

        rsd_list = []

        for b, cols in qc_cols_by_batch.items():
            dfb = data[cols].copy()

            if ignore_zero_qc:
                dfb = dfb.mask(dfb <= 0, np.nan)

            n_nonmiss = dfb.notna().sum(axis=1)
            valid = n_nonmiss >= min_qc_n

            meanb = dfb.mean(axis=1, skipna=True)
            stdb = dfb.std(axis=1, skipna=True)

            rsd_b = (stdb / meanb.replace(0, np.nan)) * 100
            rsd_b = rsd_b.where(valid, np.nan)

            variable_metadata[f"QC_RSD_{b}"] = rsd_b
            rsd_list.append(rsd_b.to_numpy())

        if len(rsd_list) > 0:
            rsd_matrix = np.column_stack(rsd_list)
            qc_rsd = np.nanmedian(rsd_matrix, axis=1)
        else:
            qc_rsd = np.full(len(data), np.nan)

    # FINAL FILTERING ---------------------------------------------------

    variable_metadata["QC_RSD"] = qc_rsd

    over_threshold = np.isfinite(qc_rsd) & (qc_rsd > rsd_threshold)

    deleted_data = data[over_threshold].copy()
    deleted_data.reset_index(drop=True, inplace=True)

    removed_ids = (
        deleted_data["cpdID"].tolist()
        if "cpdID" in deleted_data.columns
        else deleted_data.index.tolist()
    )

    removed_count = len(removed_ids)

    dropped_features_folder = _get_dropped_features_folder(state)

    txt_path = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name="removed_features_relative_standard_deviation_"
        + str(int(rsd_threshold)),
        lines=removed_ids,
    )

    _add_artifact_if_available(
        state,
        path=txt_path,
        artifact_type="dropped_features",
        description="Features removed by relative standard deviation filter.",
    )

    data = data[~over_threshold].copy()
    data.reset_index(drop=True, inplace=True)

    # PLOTTING ----------------------------------------------------------

    images = []

    if to_plot is False:
        number_plotted = 0
    elif to_plot is True:
        number_plotted = 4
    elif isinstance(to_plot, int):
        number_plotted = int(to_plot)
    else:
        raise ValueError("to_plot has to be either boolean or an integer.")

    if number_plotted > 0:
        figures_folder = os.path.join(getattr(state, "main_folder", "."), "figures")
        os.makedirs(figures_folder, exist_ok=True)

        suffixes = getattr(state, "suffixes", [".png"])

        is_qc_sample = [col in QC_samples for col in sample_cols]
        indexes = deleted_data.index.tolist()

        if len(indexes) < number_plotted:
            number_plotted = len(indexes)

        if len(indexes) == 0:
            print("No compounds with RSD > " + str(rsd_threshold) + " were found.")
        else:
            for i in range(number_plotted):
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    range(len(deleted_data.iloc[i, 1:][is_qc_sample])),
                    deleted_data.iloc[i, 1:][is_qc_sample],
                    label=str(deleted_data.iloc[i, 0]),
                    s=10,
                    alpha=0.5,
                )
                plt.xlabel("Samples in order")
                plt.ylabel("Peak Area")
                plt.title("High RSD compound: cpdID = " + str(deleted_data.iloc[i, 0]))

                base_path = os.path.join(
                    figures_folder,
                    "QC_samples_scatter_" + str(i) + "_high_RSD-deleted_by_correction",
                )

                for suffix in suffixes:
                    plt.savefig(
                        base_path + suffix,
                        dpi=400,
                        bbox_inches="tight",
                    )

                images.append(base_path + ".png")
                plt.close()

    # UPDATE STATE ------------------------------------------------------

    state.data = data
    state.variable_metadata = _filter_match_variable_metadata(data, variable_metadata)

    print("Number of features removed: " + str(removed_count))

    # REPORTING ---------------------------------------------------------

    text0 = (
        "Features with RSD% over the threshold ("
        + str(rsd_threshold)
        + ") were removed. Number of features removed: "
        + str(removed_count)
    )

    if removed_count > 0 and removed_count < 25:
        text1 = " ;being: " + str(removed_ids)
    elif removed_count >= 25:
        text1 = "The list of removed features is long and saved in: " + txt_path
    else:
        text1 = ""

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                "line",
            ]
        )

    return {
        "features_after": int(state.data.shape[0]),
        "removed_count": removed_count,
        "rsd_threshold": rsd_threshold,
        "by_batch": by_batch,
        "ignore_zero_qc": ignore_zero_qc,
        "removed_features_file": txt_path,
        "images": images,
        "skipped": False,
    }


@register_operation(
    id="filter_dilution_series_linearity",
    label="Filter Dilution Series Linearity",
    description=(
        "Filter out features whose dilution-series linearity is below the selected R² threshold."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="number_of_series",
            type="int",
            required=True,
            default=None,
            label="Number of dilution series",
            help="Number of dilution series used.",
            example="e.g.: 1, 2, ...",
        ),
        ParameterDef(
            name="threshold",
            type="float",
            required=False,
            default=0.8,
            label="R² threshold",
            help="Minimum required dilution-series R².",
        ),
        ParameterDef(
            name="which_to_take",
            type="str",
            required=False,
            default="first",
            label="Which R² to use",
            help="Choose from: best, worst, first, last, mean, median.",
        ),
        ParameterDef(
            name="concentrations",
            type="list_or_bool",
            required=False,
            default=False,
            label="Concentrations",
            help="False uses sample order. Otherwise provide concentrations as a list.",
        ),
        ParameterDef(
            name="to_plot",
            type="bool_or_int",
            required=False,
            default=False,
            label="Plot examples",
            help="False disables plotting, True plots 4 examples, integer plots that many.",
        ),
    ],
    requires=["data", "variable_metadata", "dilution_series_samples"],
    produces=["data", "variable_metadata"],
)
def filter_dilution_series_linearity(
    state: WorkflowState,
    number_of_series: int,
    threshold: float = 0.8,
    which_to_take: str = "first",
    concentrations=False,
    to_plot=False,
):
    data = state.data
    variable_metadata = state.variable_metadata
    report = state.report
    dilution_series_samples = state.dilution_series_samples

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if variable_metadata is None:
        raise ValueError("No variable metadata found in state.variable_metadata.")

    if number_of_series <= 0:
        raise ValueError("number_of_series must be greater than 0.")

    allowed_which = ["best", "worst", "first", "last", "mean", "median"]
    if which_to_take not in allowed_which:
        raise ValueError(
            "which_to_take not recognized. Choose from: "
            "'best', 'worst', 'first', 'last', 'mean' or 'median'."
        )

    if type(concentrations) != list and concentrations is not False:
        concentrations = getattr(state, "dil_concentrations", None)

        if concentrations is None:
            raise ValueError(
                "concentrations was not provided and state.dil_concentrations is not set."
            )

    if dilution_series_samples is None:
        raise ValueError("Dilution series was not defined. Define it first.")

    if dilution_series_samples is False:
        raise ValueError(
            "Dilution series were already removed. "
            "Cannot filter features by dilution series linearity."
        )

    is_dilution_series_sample = [
        True if col in dilution_series_samples else False for col in data.columns[1:]
    ]

    dilution_cols = data.columns[1:][is_dilution_series_sample]

    if len(dilution_cols) == 0:
        raise ValueError(
            "No dilution series sample columns found in data. "
            "Check state.dilution_series_samples."
        )

    r2 = []
    all_series_pred_y = []
    all_series_x = []
    all_series_y = []

    # CALCULATE R2 ------------------------------------------------------

    for i in range(len(data)):
        y = np.array(data.iloc[i, 1:][is_dilution_series_sample].dropna().values)
        original_length = len(y)

        if len(y) == 0:
            r2.append(np.nan)
            all_series_pred_y.append([])
            all_series_x.append([])
            all_series_y.append([])
            continue

        if len(y) % number_of_series != 0:
            raise ValueError(
                "The number of dilution-series values is not divisible by "
                "number_of_series. Check dilution series sample definitions."
            )

        y = y.reshape((number_of_series, -1))

        is_length_of_one = True

        if concentrations:
            x = np.array(concentrations)

            if len(x) != len(y[0]):
                if len(x) == original_length:
                    x = x.reshape((number_of_series, -1))
                    is_length_of_one = False
                else:
                    raise ValueError(
                        "The length of concentrations is not the same as the number "
                        "of samples in each series or all series together."
                    )
        else:
            x = np.arange(len(y[0]))

        this_series_r2 = []
        this_series_pred_y = []
        this_series_x = []

        for number, series in enumerate(y):
            if is_length_of_one:
                this_x = x
            else:
                this_x = x[number]

            this_x = np.array(this_x, dtype=float)
            series = np.array(series, dtype=float)

            A = np.vstack([this_x, np.ones(len(this_x))]).T
            slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
            y_pred = slope * this_x + intercept

            this_series_pred_y.append(y_pred)
            this_series_x.append(this_x)

            ss_res = np.sum((series - y_pred) ** 2)
            ss_tot = np.sum((series - np.mean(series)) ** 2)

            if ss_tot == 0:
                r_squared = np.nan
            else:
                r_squared = 1 - (ss_res / ss_tot)

            this_series_r2.append(r_squared)

        all_series_pred_y.append(this_series_pred_y)
        all_series_x.append(this_series_x)
        all_series_y.append(y)

        finite_r2 = [x for x in this_series_r2 if np.isfinite(x)]

        if len(finite_r2) == 0:
            r2.append(np.nan)
        elif which_to_take == "best":
            r2.append(max(finite_r2))
        elif which_to_take == "worst":
            r2.append(min(finite_r2))
        elif which_to_take == "first":
            r2.append(this_series_r2[0])
        elif which_to_take == "last":
            r2.append(this_series_r2[-1])
        elif which_to_take == "mean":
            r2.append(np.nanmean(this_series_r2))
        elif which_to_take == "median":
            r2.append(np.nanmedian(this_series_r2))

    r2 = pd.Series(r2, index=data.index)

    variable_metadata["Dilution_Series_R2"] = r2

    # PLOTTING ----------------------------------------------------------

    images = []

    if to_plot:
        if to_plot is True:
            number_plotted = 4
        elif isinstance(to_plot, int):
            number_plotted = int(to_plot)
        else:
            raise ValueError("to_plot has to be either True/False or an integer.")

        figures_folder = os.path.join(getattr(state, "main_folder", "."), "figures")
        os.makedirs(figures_folder, exist_ok=True)

        suffixes = getattr(state, "suffixes", [".png"])

        indexes = r2[r2 < threshold].index.tolist()[:number_plotted]

        if len(indexes) == 0:
            print("No compounds with R2 < " + str(threshold) + " were found.")
        else:
            for i in indexes:
                plt.figure()

                for series, y_pred, this_x in zip(
                    all_series_y[i],
                    all_series_pred_y[i],
                    all_series_x[i],
                ):
                    plt.scatter(this_x, series, alpha=0.5, marker="o")
                    plt.plot(this_x, y_pred, alpha=0.5)

                plt.xlabel("Dilution factor")
                plt.ylabel("Peak Area")
                plt.title(
                    "Dilution series - LOW linearity compound: cpdID = "
                    + str(data.iloc[i, 0])
                    + " ;R2 = "
                    + str(r2[i])[:7]
                    + "("
                    + which_to_take
                    + ")"
                )

                base_path = os.path.join(
                    figures_folder,
                    "dilution_series_linearity_"
                    + str(i)
                    + "_low_R2-deleted_by_correction",
                )

                for suffix in suffixes:
                    plt.savefig(
                        base_path + suffix,
                        dpi=400,
                        bbox_inches="tight",
                    )

                images.append(base_path + ".png")
                plt.close()

    # FILTERING ---------------------------------------------------------

    under_threshold = r2 < threshold

    removed_ids = (
        data.loc[under_threshold, "cpdID"].tolist()
        if "cpdID" in data.columns
        else list(np.where(under_threshold)[0])
    )

    removed_count = len(removed_ids)

    dropped_features_folder = _get_dropped_features_folder(state)

    txt_path = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name=(
            "removed_features_dilution_series_linearity_"
            + str(int(threshold * 100))
            + "pct"
        ),
        lines=removed_ids,
    )

    _add_artifact_if_available(
        state,
        path=txt_path,
        artifact_type="dropped_features",
        description="Features removed by dilution-series linearity filter.",
    )

    data = data[~under_threshold].reset_index(drop=True)

    state.data = data
    state.variable_metadata = _filter_match_variable_metadata(data, variable_metadata)

    print(f"Number of features removed: {removed_count}")

    # REPORTING ---------------------------------------------------------

    text0 = (
        "Features with dilution series linearity (R2) under the threshold ("
        + str(threshold)
        + ") were removed. Number of features removed: "
        + str(removed_count)
    )

    if removed_count > 0 and removed_count < 25:
        text1 = " ;being: " + str(removed_ids)
    elif removed_count >= 25:
        text1 = "The list of removed features is long and saved in: " + txt_path
    else:
        text1 = ""

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                "line",
            ]
        )

    return {
        "features_after": int(state.data.shape[0]),
        "removed_count": removed_count,
        "threshold": threshold,
        "which_to_take": which_to_take,
        "removed_features_file": txt_path,
        "images": images,
    }


@register_operation(
    id="filter_number_of_corrected_batches",
    label="Filter Number of Corrected Batches",
    description=(
        "Filter out features corrected in fewer batches than the selected threshold."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="threshold",
            type="float",
            required=False,
            default=0.8,
            label="Corrected batch threshold",
            help=(
                "If below 1, interpreted as fraction of all batches. "
                "If 1 or above, interpreted as minimum number of corrected batches."
            ),
        ),
    ],
    requires=["data", "variable_metadata", "batch_info"],
    produces=["data", "variable_metadata"],
)
def filter_number_of_corrected_batches(
    state: WorkflowState,
    threshold: float = 0.8,
):
    data = state.data
    variable_metadata = state.variable_metadata
    report = state.report
    batch_info = state.batch_info

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if variable_metadata is None:
        raise ValueError("No variable metadata found in state.variable_metadata.")

    if batch_info is None:
        raise ValueError("No batch information found in state.batch_info.")

    if "Batch" not in batch_info.columns:
        raise ValueError("state.batch_info must contain a 'Batch' column.")

    if "corrected_batches" not in variable_metadata.columns:
        raise ValueError(
            "state.variable_metadata must contain a 'corrected_batches' column."
        )

    if "cpdID" not in data.columns or "cpdID" not in variable_metadata.columns:
        raise ValueError("Both data and variable_metadata must contain 'cpdID'.")

    start_n = len(data)

    nm_of_batches = len(batch_info["Batch"].dropna().unique())

    if threshold < 1:
        percentage = True
        threshold_used = int(threshold * nm_of_batches)
    else:
        percentage = False
        threshold_used = threshold

    corrected_batches_dict = {
        cpdID: corrected_batches
        for cpdID, corrected_batches in zip(
            variable_metadata["cpdID"],
            variable_metadata["corrected_batches"],
        )
    }

    keep_mask = data["cpdID"].map(corrected_batches_dict).fillna(0) >= threshold_used

    removed = data.loc[~keep_mask, "cpdID"].tolist()

    data = data.loc[keep_mask].copy()
    data.reset_index(drop=True, inplace=True)

    removed_count = start_n - len(data)

    dropped_features_folder = _get_dropped_features_folder(state)

    txt_path = _write_versioned_txt(
        folder=dropped_features_folder,
        base_name="removed_features_number_of_corrected_batches_"
        + str(int(threshold_used)),
        lines=removed,
    )

    _add_artifact_if_available(
        state,
        path=txt_path,
        artifact_type="dropped_features",
        description="Features removed by number of corrected batches filter.",
    )

    variable_metadata = variable_metadata[
        variable_metadata["cpdID"].isin(data["cpdID"])
    ].copy()
    variable_metadata.reset_index(drop=True, inplace=True)

    state.data = data
    state.variable_metadata = _filter_match_variable_metadata(data, variable_metadata)

    print(f"Number of features removed: {removed_count}")

    # REPORTING ---------------------------------------------------------

    text0 = (
        "Features that were corrected in less than the threshold number of batches ("
        + str(threshold_used)
        + (
            " (" + str(int(threshold_used / nm_of_batches * 100)) + "%)"
            if percentage
            else ""
        )
        + ") were removed. Number of features removed: "
        + str(removed_count)
    )

    if removed_count > 0 and removed_count < 25:
        text1 = " ;being: " + str(removed)
    elif removed_count >= 25:
        text1 = "The list of removed features is long and saved in: " + txt_path
    else:
        text1 = ""

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                "line",
            ]
        )

    return {
        "features_after": int(state.data.shape[0]),
        "removed_count": removed_count,
        "threshold_input": threshold,
        "threshold_used": threshold_used,
        "removed_features_file": txt_path,
    }


@register_operation(
    id="drop_samples",
    label="Drop Samples",
    description="Drop selected sample columns from the data matrix.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="column_indexes_to_drop",
            type="list",
            required=True,
            default=None,
            label="Column indexes to drop",
            help="List of column indexes to remove.",
            example="e.g.: [1, 2, 3]",
        ),
        ParameterDef(
            name="cpdID_as_zero",
            type="bool",
            required=False,
            default=True,
            label="cpdID as index 0",
            help="If True, cpdID is counted as column index 0.",
        ),
    ],
    requires=["data"],
    produces=["data", "metadata", "batch_info", "batch", "variable_metadata"],
)
def drop_samples(
    state: WorkflowState,
    column_indexes_to_drop,
    cpdID_as_zero: bool = True,
):
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if not isinstance(column_indexes_to_drop, (list, tuple, np.ndarray, pd.Series)):
        column_indexes_to_drop = [column_indexes_to_drop]

    column_indexes_to_drop = [int(i) for i in column_indexes_to_drop]

    if cpdID_as_zero and 0 in column_indexes_to_drop:
        raise ValueError(
            "cpdID cannot be dropped. Indexes cannot include 0 when cpdID_as_zero is True."
        )

    bad = [i for i in column_indexes_to_drop if i < 0 or i >= len(data.columns)]
    if bad:
        raise ValueError(f"Some column indexes are out of range: {bad}")

    dropped_column_names = data.columns[column_indexes_to_drop].tolist()

    data = data.drop(data.columns[column_indexes_to_drop], axis=1)

    sample_indexes_to_drop = column_indexes_to_drop.copy()

    if cpdID_as_zero:
        sample_indexes_to_drop = [i - 1 for i in column_indexes_to_drop]

    metadata = getattr(state, "metadata", None)
    batch = getattr(state, "batch", None)
    batch_info = getattr(state, "batch_info", None)

    if metadata is not None:
        metadata = metadata.drop(metadata.index[sample_indexes_to_drop])
        metadata.reset_index(drop=True, inplace=True)

    if batch_info is not None:
        batch_info = batch_info.drop(batch_info.index[sample_indexes_to_drop])
        batch_info.reset_index(drop=True, inplace=True)

    if batch is not None:
        batch = [batch[i] for i in range(len(batch)) if i not in sample_indexes_to_drop]

    variable_metadata = _filter_match_variable_metadata(data, state.variable_metadata)

    state.data = data
    state.metadata = metadata
    state.batch_info = batch_info
    state.batch = batch
    state.variable_metadata = variable_metadata

    removed_count = len(column_indexes_to_drop)

    print(
        "Specified samples: "
        + str(column_indexes_to_drop)
        + " were removed from the data."
    )

    text0 = "Specified samples were removed from the data: " + str(dropped_column_names)

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                "line",
            ]
        )

    return {
        "dropped_sample_indexes": column_indexes_to_drop,
        "dropped_sample_names": dropped_column_names,
        "removed_count": removed_count,
    }


@register_operation(
    id="drop_features",
    label="Drop Features",
    description="Drop selected feature rows by row index.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="row_indexes_to_drop",
            type="list",
            required=True,
            default=None,
            label="Row indexes to drop",
            help="List of feature row indexes to remove. Indexing starts at 0.",
            example="e.g.: [1, 2, 3]",
        ),
        ParameterDef(
            name="note",
            type="str",
            required=False,
            default="",
            label="Note",
            help="Optional explanation added to the report.",
        ),
    ],
    requires=["data"],
    produces=["data", "variable_metadata"],
)
def drop_features(
    state: WorkflowState,
    row_indexes_to_drop,
    note: str = "",
):
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if not isinstance(row_indexes_to_drop, (list, np.ndarray, pd.Series)):
        row_indexes_to_drop = [row_indexes_to_drop]

    row_indexes_to_drop = [int(i) for i in row_indexes_to_drop]

    bad = [i for i in row_indexes_to_drop if i < 0 or i >= len(data)]
    if bad:
        raise ValueError(f"Some row indexes are out of range: {bad}")

    removed_ids = (
        data.iloc[row_indexes_to_drop]["cpdID"].tolist()
        if "cpdID" in data.columns
        else row_indexes_to_drop
    )

    data = data.drop(data.index[row_indexes_to_drop]).reset_index(drop=True)
    variable_metadata = _filter_match_variable_metadata(data, state.variable_metadata)

    state.data = data
    state.variable_metadata = variable_metadata

    note = str(note)

    print(
        "Specified features with indexes: "
        + str(row_indexes_to_drop)
        + " were removed from the data."
    )
    print("Note: " + note)

    text0 = (
        "Specified features with indexes: "
        + str(row_indexes_to_drop)
        + " were removed from the data."
    )
    text1 = note

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                "line",
            ]
        )

    return {
        "dropped_feature_indexes": row_indexes_to_drop,
        "dropped_feature_ids": removed_ids,
        "removed_count": len(row_indexes_to_drop),
        "note": note,
    }


@register_operation(
    id="drop_features_by_cpdID",
    label="Drop Features by cpdID",
    description="Drop selected feature rows by cpdID.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="cpdIDs_to_drop",
            type="list",
            required=True,
            default=None,
            label="cpdIDs to drop",
            help="List of cpdIDs to remove.",
            example="e.g.: [M232.88324-T24.387, M232.89504-T24.390]",
        ),
        ParameterDef(
            name="note",
            type="str",
            required=False,
            default="",
            label="Note",
            help="Optional explanation added to the report.",
        ),
    ],
    requires=["data"],
    produces=["data", "variable_metadata"],
)
def drop_features_by_cpdID(
    state: WorkflowState,
    cpdIDs_to_drop,
    note: str = "",
):
    data = state.data
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if "cpdID" not in data.columns:
        raise ValueError("No cpdID column in the data.")

    if isinstance(cpdIDs_to_drop, str):
        cpdIDs_to_drop = [cpdIDs_to_drop]

    cpdIDs_to_drop = [str(x) for x in cpdIDs_to_drop]

    existing_ids = data["cpdID"].astype(str)
    removed_ids = data.loc[existing_ids.isin(cpdIDs_to_drop), "cpdID"].tolist()

    data = data[~existing_ids.isin(cpdIDs_to_drop)].reset_index(drop=True)
    variable_metadata = _filter_match_variable_metadata(data, state.variable_metadata)

    state.data = data
    state.variable_metadata = variable_metadata

    note = str(note)

    print("Specified features: " + str(cpdIDs_to_drop) + " were removed from the data.")
    print("Note: " + note)

    text0 = (
        "Specified features: " + str(cpdIDs_to_drop) + " were removed from the data."
    )
    text1 = note

    if report is not None:
        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                "line",
            ]
        )

    return {
        "requested_cpdIDs": cpdIDs_to_drop,
        "dropped_feature_ids": removed_ids,
        "removed_count": len(removed_ids),
        "note": note,
    }


@register_operation(
    id="drop_samples_by_metadata",
    label="Drop Samples by Metadata",
    description="Drop samples whose metadata column has a selected value.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="metadata_column",
            type="str",
            required=True,
            default=None,
            label="Metadata column",
            help="Name of the metadata column used for selecting samples.",
            example="e.g.: Diagnosis",
        ),
        ParameterDef(
            name="value",
            type="str",
            required=True,
            default=None,
            label="Value",
            help="Value in the metadata column used for selecting samples.",
            example="e.g.: Suspected",
        ),
    ],
    requires=["data", "metadata"],
    produces=["data", "metadata", "batch_info", "batch", "variable_metadata"],
)
def drop_samples_by_metadata(
    state: WorkflowState,
    metadata_column: str,
    value,
):
    metadata = state.metadata

    if metadata is None:
        raise ValueError("No metadata found in state.metadata.")

    if metadata_column not in metadata.columns:
        raise ValueError(f"Metadata column not found: {metadata_column}")

    column_indexes_to_drop = metadata[metadata[metadata_column] == value].index.tolist()

    # Add 1 because the first data column is cpdID.
    column_indexes_to_drop = [i + 1 for i in column_indexes_to_drop]

    print(column_indexes_to_drop)

    return drop_samples(
        state,
        column_indexes_to_drop=column_indexes_to_drop,
        cpdID_as_zero=True,
    )


@register_operation(
    id="drop_blank_samples",
    label="Drop Blank Samples",
    description="Drop blank sample columns from the data matrix.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[],
    requires=["data", "blank_samples"],
    produces=[
        "data",
        "metadata",
        "batch_info",
        "batch",
        "variable_metadata",
        "blank_samples",
    ],
)
def drop_blank_samples(state: WorkflowState):
    if state.blank_samples is None:
        raise ValueError("No blank samples were defined.")

    if state.blank_samples is False:
        raise ValueError("Blank samples were already removed.")

    blank_samples = state.blank_samples

    data = state.data
    if data is None:
        raise ValueError("No data loaded in state.data.")

    blank_indexes = [
        data.columns.get_loc(col) for col in blank_samples if col in data.columns
    ]

    result = drop_samples(
        state,
        column_indexes_to_drop=blank_indexes,
        cpdID_as_zero=True,
    )

    state.blank_samples = False

    print("Blank samples were removed from the data.")

    if state.report is not None:
        state.report.add_together(
            [
                ("text", "Blank samples were removed from the data."),
                "line",
            ]
        )

    result["dropped_group"] = "blank_samples"
    return result


@register_operation(
    id="drop_dilution_series_samples",
    label="Drop Dilution Series Samples",
    description="Drop dilution-series sample columns from the data matrix.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[],
    requires=["data", "dilution_series_samples"],
    produces=[
        "data",
        "metadata",
        "batch_info",
        "batch",
        "variable_metadata",
        "dilution_series_samples",
    ],
)
def drop_dilution_series_samples(state: WorkflowState):
    if state.dilution_series_samples is None:
        raise ValueError("No dilution series were defined.")

    if state.dilution_series_samples is False:
        raise ValueError("Dilution series were already removed.")

    dilution_series_samples = state.dilution_series_samples

    data = state.data
    if data is None:
        raise ValueError("No data loaded in state.data.")

    dilution_series_indexes = []

    for col in dilution_series_samples:
        if col in data.columns:
            dilution_series_indexes.append(data.columns.get_loc(col))
        else:
            print(f"Column {col} not found, probably already deleted.")

    if not dilution_series_indexes:
        print("No dilution series samples were found to remove.")

        return {
            "removed_count": 0,
            "dropped_sample_indexes": [],
            "dropped_group": "dilution_series_samples",
        }

    result = drop_samples(
        state,
        column_indexes_to_drop=dilution_series_indexes,
        cpdID_as_zero=True,
    )

    state.dilution_series_samples = False

    print("Dilution series samples were removed from the data.")

    if state.report is not None:
        state.report.add_together(
            [
                ("text", "Dilution series samples were removed from the data."),
                "line",
            ]
        )

    result["dropped_group"] = "dilution_series_samples"
    return result


@register_operation(
    id="drop_standard_samples",
    label="Drop Standard Samples",
    description="Drop standard sample columns from the data matrix.",
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[],
    requires=["data", "standard_samples"],
    produces=[
        "data",
        "metadata",
        "batch_info",
        "batch",
        "variable_metadata",
        "standard_samples",
    ],
)
def drop_standard_samples(state: WorkflowState):
    if state.standard_samples is None:
        raise ValueError("No standards were defined.")

    if state.standard_samples is False:
        raise ValueError("Standards were already removed.")

    standard_samples = state.standard_samples

    data = state.data
    if data is None:
        raise ValueError("No data loaded in state.data.")

    standard_indexes = []

    for col in standard_samples:
        if col in data.columns:
            standard_indexes.append(data.columns.get_loc(col))
        else:
            print(f"Column {col} not found, probably already deleted.")

    if not standard_indexes:
        print("No standards were found to remove.")

        return {
            "removed_count": 0,
            "dropped_sample_indexes": [],
            "dropped_group": "standard_samples",
        }

    result = drop_samples(
        state,
        column_indexes_to_drop=standard_indexes,
        cpdID_as_zero=True,
    )

    state.standard_samples = False

    print("Standards were removed from the data.")

    if state.report is not None:
        state.report.add_together(
            [
                ("text", "Standards were removed from the data."),
                "line",
            ]
        )

    result["dropped_group"] = "standard_samples"
    return result


@register_operation(
    id="filter_sparse_spike_features",
    label="Sparse/Spike Feature Filter",
    description=(
        "Experimental diagnostic filter for sparse/spiky low-prevalence features. "
        "By default it only flags features and adds diagnostics to variable metadata. "
        "For example metabolites of a specific drug, they carry most of the signal in PCA and othe statistics."
        "If remove=True, flagged features are removed from the data."
    ),
    citation="",
    category_tags=[OperationTag.FILTER],
    parameter_schema=[
        ParameterDef(
            name="class_column",
            type="str",
            required=False,
            default="Class",
            label="Class column",
            help="Metadata column used to identify QC, Blank, NIST, Standard, Dilution, etc.",
        ),
        ParameterDef(
            name="group_column",
            type="str",
            required=False,
            default="Diagnosis",
            label="Group column",
            help="Biological grouping column. Used to protect real group-specific features.",
        ),
        ParameterDef(
            name="exclude_classes",
            type="list",
            required=False,
            default=["QC", "Blank", "NIST", "Standard", "Dilution", "dQC"],
            label="Excluded classes",
            help="Classes excluded from biological-sample prevalence calculation.",
        ),
        ParameterDef(
            name="presence_mode",
            type="str",
            required=False,
            default="qc_fraction",
            label="Presence mode",
            help="Choose 'qc_fraction' or 'zero'.",
        ),
        ParameterDef(
            name="qc_fraction",
            type="float",
            required=False,
            default=0.05,
            label="QC fraction",
            help="Fraction of QC median used as feature-specific detection threshold.",
        ),
        ParameterDef(
            name="absolute_min",
            type="float",
            required=False,
            default=0,
            label="Absolute minimum",
            help="Minimum intensity threshold.",
        ),
        ParameterDef(
            name="min_bio_presence",
            type="float",
            required=False,
            default=0.15,
            label="Minimum biological presence",
            help="Minimum fraction of all biological samples where feature should be present.",
        ),
        ParameterDef(
            name="min_group_presence",
            type="float",
            required=False,
            default=0.40,
            label="Minimum group presence",
            help="Minimum fraction of at least one biological group where feature should be present.",
        ),
        ParameterDef(
            name="max_n_present",
            type="int",
            required=False,
            default=5,
            label="Maximum present samples",
            help="Flag features present in this many biological samples or fewer.",
        ),
        ParameterDef(
            name="top_k",
            type="int",
            required=False,
            default=3,
            label="Top k samples",
            help="Number of highest biological samples used for signal dominance check.",
        ),
        ParameterDef(
            name="top_k_share_threshold",
            type="float",
            required=False,
            default=0.60,
            label="Top-k share threshold",
            help="Flag if top-k samples explain more than this fraction of total biological signal.",
        ),
        ParameterDef(
            name="max_to_median_threshold",
            type="float",
            required=False,
            default=20,
            label="Max-to-median threshold",
            help="Flag if max biological signal divided by median present biological signal is very high.",
        ),
        ParameterDef(
            name="qc_dominance_ratio",
            type="float",
            required=False,
            default=10,
            label="QC dominance ratio",
            help="Flag QC-dominant features when QC median divided by biological present median is high.",
        ),
        ParameterDef(
            name="remove",
            type="bool",
            required=False,
            default=False,
            label="Remove flagged features",
            help="If False, only diagnostics are added. If True, flagged features are removed.",
        ),
        ParameterDef(
            name="note",
            type="str",
            required=False,
            default="sparse/spiky low-prevalence feature filter",
            label="Note",
            help="Note added to the report.",
        ),
    ],
    requires=["data", "variable_metadata", "metadata"],
    produces=["data", "variable_metadata", "sparse_spike_diagnostics"],
)
def filter_sparse_spike_features(
    state: WorkflowState,
    class_column: str = "Class",
    group_column: str = "Diagnosis",
    exclude_classes=None,
    presence_mode: str = "qc_fraction",
    qc_fraction: float = 0.05,
    absolute_min: float = 0,
    min_bio_presence: float = 0.15,
    min_group_presence: float = 0.40,
    max_n_present: int = 5,
    top_k: int = 3,
    top_k_share_threshold: float = 0.60,
    max_to_median_threshold: float = 20,
    qc_dominance_ratio: float = 10,
    remove: bool = False,
    note: str = "sparse/spiky low-prevalence feature filter",
):
    """
    EXPERIMENTAL DIAGNOSTIC FILTER

    Flags or removes features where only a small number of biological samples
    carry most of the signal, while most biological samples are near zero.

    This is intended mainly as a sensitivity-analysis filter before PCA/statistics.
    It is not a blank-contamination filter.

    By default:
        remove=False

    Therefore the operation only adds diagnostic columns to variable_metadata.
    Actual feature removal happens only when remove=True.
    """

    data = state.data.copy() if state.data is not None else None
    variable_metadata = (
        state.variable_metadata.copy() if state.variable_metadata is not None else None
    )
    metadata = state.metadata.copy() if state.metadata is not None else None
    report = state.report

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if variable_metadata is None:
        raise ValueError("No variable metadata found in state.variable_metadata.")

    if metadata is None:
        raise ValueError("No metadata found in state.metadata.")

    if exclude_classes is None:
        exclude_classes = ["QC", "Blank", "NIST", "Standard", "Dilution", "dQC"]

    if isinstance(exclude_classes, str):
        exclude_classes = [exclude_classes]

    exclude_classes = list(exclude_classes)

    if "cpdID" not in data.columns:
        raise ValueError("Expected 'cpdID' column in state.data.")

    if "cpdID" not in variable_metadata.columns:
        raise ValueError("Expected 'cpdID' column in state.variable_metadata.")

    if "Sample File" not in metadata.columns:
        raise ValueError("Expected 'Sample File' column in state.metadata.")

    if class_column not in metadata.columns:
        raise ValueError(f"{class_column} not found in metadata.")

    if group_column not in metadata.columns:
        raise ValueError(f"{group_column} not found in metadata.")

    if presence_mode not in ["qc_fraction", "zero"]:
        raise ValueError("presence_mode must be 'qc_fraction' or 'zero'.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if max_n_present < 0:
        raise ValueError("max_n_present cannot be negative.")

    # Data matrix with cpdID as index.
    X = data.set_index("cpdID").apply(pd.to_numeric, errors="coerce")
    sample_cols = list(X.columns)

    # QC columns.
    qc_samples = state.QC_samples if isinstance(state.QC_samples, list) else []
    qc_cols = [c for c in sample_cols if c in qc_samples]

    # Biological columns based on metadata classes.
    meta_present = metadata[metadata["Sample File"].isin(sample_cols)].copy()

    bio_meta = meta_present[
        ~meta_present[class_column].astype(str).isin(exclude_classes)
    ].copy()

    bio_cols = bio_meta["Sample File"].tolist()
    bio_cols = [c for c in bio_cols if c in sample_cols]

    if len(bio_cols) == 0:
        raise ValueError("No biological sample columns found.")

    X_bio = X[bio_cols]

    # ------------------------------------------------------------
    # Define feature-specific presence threshold
    # ------------------------------------------------------------

    if presence_mode == "qc_fraction":
        if len(qc_cols) == 0:
            raise ValueError(
                "No QC columns found, cannot use presence_mode='qc_fraction'. "
                "Use presence_mode='zero' or define QC samples."
            )

        qc_median = X[qc_cols].replace(0, np.nan).median(axis=1)
        presence_cutoff = qc_fraction * qc_median
        presence_cutoff = presence_cutoff.fillna(absolute_min)
        presence_cutoff = presence_cutoff.clip(lower=absolute_min)

    elif presence_mode == "zero":
        qc_median = pd.Series(np.nan, index=X.index)
        presence_cutoff = pd.Series(absolute_min, index=X.index)

    # ------------------------------------------------------------
    # Presence in biological samples
    # ------------------------------------------------------------

    present_bio = X_bio.gt(presence_cutoff, axis=0)
    n_present = present_bio.sum(axis=1)
    bio_presence_frac = present_bio.mean(axis=1)

    # ------------------------------------------------------------
    # Group-wise presence
    # Protects real group-specific features.
    # ------------------------------------------------------------

    group_presence = pd.DataFrame(index=X.index)

    for group in bio_meta[group_column].dropna().unique():
        group_cols = bio_meta.loc[
            bio_meta[group_column] == group,
            "Sample File",
        ].tolist()

        group_cols = [c for c in group_cols if c in X_bio.columns]

        if len(group_cols) == 0:
            continue

        safe_group_name = str(group)

        group_presence[safe_group_name] = (
            X[group_cols]
            .gt(
                presence_cutoff,
                axis=0,
            )
            .mean(axis=1)
        )

    if group_presence.shape[1] > 0:
        max_group_presence = group_presence.max(axis=1)
    else:
        max_group_presence = pd.Series(0, index=X.index)

    # ------------------------------------------------------------
    # Signal dominance by top-k biological samples
    # ------------------------------------------------------------

    X_bio_signal = X_bio.where(X_bio.gt(presence_cutoff, axis=0), 0)

    vals = X_bio_signal.to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)

    total_signal = vals.sum(axis=1)
    sorted_vals = np.sort(vals, axis=1)[:, ::-1]
    top_k_sum = sorted_vals[:, : min(top_k, sorted_vals.shape[1])].sum(axis=1)

    top_k_share = np.divide(
        top_k_sum,
        total_signal,
        out=np.zeros_like(top_k_sum, dtype=float),
        where=total_signal > 0,
    )

    top_k_share = pd.Series(top_k_share, index=X.index)

    # ------------------------------------------------------------
    # Max / median among present biological samples
    # ------------------------------------------------------------

    X_bio_present = X_bio.where(X_bio.gt(presence_cutoff, axis=0), np.nan)

    bio_present_median = X_bio_present.median(axis=1)
    bio_max = X_bio_present.max(axis=1)

    max_to_median = bio_max / bio_present_median.replace(0, np.nan)

    # ------------------------------------------------------------
    # QC dominance
    # ------------------------------------------------------------

    qc_to_bio_present_median = qc_median / bio_present_median.replace(0, np.nan)

    # ------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------

    low_prevalence = (bio_presence_frac < min_bio_presence) | (
        n_present <= max_n_present
    )

    not_group_consistent = max_group_presence < min_group_presence

    spiky_signal = (top_k_share >= top_k_share_threshold) | (
        max_to_median >= max_to_median_threshold
    )

    sparse_spike = low_prevalence & not_group_consistent & spiky_signal

    qc_dominant_sparse = (
        (qc_to_bio_present_median >= qc_dominance_ratio)
        & (bio_presence_frac < min_bio_presence)
        & not_group_consistent
    )

    flag = sparse_spike | qc_dominant_sparse

    diagnostics = pd.DataFrame(
        {
            "cpdID": X.index,
            "presence_cutoff": presence_cutoff,
            "QC_median": qc_median,
            "bio_presence_frac": bio_presence_frac,
            "n_present_bio": n_present,
            "max_group_presence": max_group_presence,
            "top_k_share": top_k_share,
            "bio_max": bio_max,
            "bio_present_median": bio_present_median,
            "max_to_median": max_to_median,
            "qc_to_bio_present_median": qc_to_bio_present_median,
            "flag_sparse_spike": sparse_spike,
            "flag_qc_dominant_sparse": qc_dominant_sparse,
            "flag_remove": flag,
        }
    ).reset_index(drop=True)

    # Add group-specific presence columns.
    for col in group_presence.columns:
        diagnostics[f"group_presence_{col}"] = group_presence[col].to_numpy()

    removed_ids = diagnostics.loc[
        diagnostics["flag_remove"],
        "cpdID",
    ].tolist()

    removed_count = len(removed_ids)

    print(f"Sparse/spike diagnostic features flagged: {removed_count}")

    # ------------------------------------------------------------
    # Store diagnostics
    # ------------------------------------------------------------

    diagnostic_columns = [
        "presence_cutoff",
        "QC_median",
        "bio_presence_frac",
        "n_present_bio",
        "max_group_presence",
        "top_k_share",
        "bio_max",
        "bio_present_median",
        "max_to_median",
        "qc_to_bio_present_median",
        "flag_sparse_spike",
        "flag_qc_dominant_sparse",
        "flag_remove",
    ]

    existing_diagnostic_columns = [
        col
        for col in variable_metadata.columns
        if col in diagnostic_columns or col.startswith("group_presence_")
    ]

    if existing_diagnostic_columns:
        variable_metadata = variable_metadata.drop(
            columns=existing_diagnostic_columns,
            errors="ignore",
        )

    variable_metadata = variable_metadata.merge(
        diagnostics,
        on="cpdID",
        how="left",
    )

    # Store full diagnostics for backend use.
    # This does not require the field to be explicitly declared unless WorkflowState uses slots.
    state.sparse_spike_diagnostics = diagnostics

    # ------------------------------------------------------------
    # Optional removal
    # ------------------------------------------------------------

    txt_path = None

    if remove:
        data = data[~data["cpdID"].isin(removed_ids)].reset_index(drop=True)

        state.data = data
        state.variable_metadata = _filter_match_variable_metadata(
            data,
            variable_metadata,
        )

        dropped_features_folder = _get_dropped_features_folder(state)

        txt_path = _write_versioned_txt(
            folder=dropped_features_folder,
            base_name="removed_features_sparse_spike",
            lines=removed_ids,
        )

        _add_artifact_if_available(
            state,
            path=txt_path,
            artifact_type="dropped_features",
            description="Features removed by sparse/spike diagnostic filter.",
        )

        if report is not None:
            report.add_together(
                [
                    (
                        "text",
                        (
                            "Features flagged as sparse/spiky were removed. "
                            f"Number removed: {removed_count}."
                        ),
                    ),
                    ("text", f"Removal note: {note}"),
                    ("text", f"Removed feature list saved to: {txt_path}"),
                    "line",
                ]
            )

    else:
        state.data = data
        state.variable_metadata = variable_metadata

        if report is not None:
            report.add_together(
                [
                    (
                        "text",
                        (
                            "Sparse/spike diagnostics were calculated. "
                            f"Number flagged: {removed_count}. "
                            "No features were removed."
                        ),
                    ),
                    ("text", f"Diagnostic note: {note}"),
                    "line",
                ]
            )

    return {
        "flagged_count": removed_count,
        "removed_count": removed_count if remove else 0,
        "remove": remove,
        "presence_mode": presence_mode,
        "qc_fraction": qc_fraction,
        "absolute_min": absolute_min,
        "min_bio_presence": min_bio_presence,
        "min_group_presence": min_group_presence,
        "max_n_present": max_n_present,
        "top_k": top_k,
        "top_k_share_threshold": top_k_share_threshold,
        "max_to_median_threshold": max_to_median_threshold,
        "qc_dominance_ratio": qc_dominance_ratio,
        "removed_features_file": txt_path,
    }
