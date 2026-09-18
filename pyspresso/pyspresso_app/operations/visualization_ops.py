# pyspresso_app/operations/visualization_ops.py

import os

import numpy as np
import pandas as pd

import ast
import re
from scipy.stats import gaussian_kde
from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState
from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def _load_plotting():
    """
    Import matplotlib lazily so operation registration does not fail
    if matplotlib is not installed yet.
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization operations. "
            "Install it with: python -m pip install matplotlib"
        ) from exc

    return plt, mpl, Line2D


def _ensure_main_folders(state: WorkflowState):
    """
    Ensure main output folder and figures folder exist.
    """
    main_folder = getattr(state, "main_folder", None)

    if main_folder is None:
        workflow_id = getattr(state, "workflow_id", "workflow")
        main_folder = os.path.join("outputs", str(workflow_id))
        state.main_folder = main_folder

    figures_folder = os.path.join(main_folder, "figures")

    os.makedirs(main_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    return main_folder, figures_folder


def _get_suffixes(state: WorkflowState):
    suffixes = getattr(state, "suffixes", None)

    if suffixes is None:
        suffixes = [".png"]
        state.suffixes = suffixes

    return suffixes


def _unique_path(path: str):
    """
    Return a filesystem path that does not yet exist by appending an index.
    """
    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{root}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _as_sample_set(value):
    """
    Convert sample lists to set. Handles None/False used in PySPRESSO.
    """
    if value is None or value is False:
        return set()

    if isinstance(value, str):
        return {value}

    return set(value)


def _resolve_feature_indices(show, data):
    """
    Resolve show parameter into feature row indexes.

    Supported:
        "default" -> 5 evenly spaced features
        "all"     -> all features
        "none"    -> no features
        int       -> one row index
        str       -> row index or cpdID
        list      -> row indexes and/or cpdIDs
    """
    n_features = len(data)

    if n_features == 0:
        return []

    cpd_ids = (
        data["cpdID"].astype(str).tolist()
        if "cpdID" in data.columns
        else [str(i) for i in range(n_features)]
    )

    if isinstance(show, str):
        if show == "default":
            return np.linspace(
                0, n_features - 1, min(5, n_features), dtype=int
            ).tolist()

        if show == "all":
            return list(range(n_features))

        if show == "none":
            return []

        try:
            idx = int(show)
            return [idx] if 0 <= idx < n_features else []
        except ValueError:
            if show in cpd_ids:
                return [cpd_ids.index(show)]
            return []

    if isinstance(show, (int, np.integer)):
        idx = int(show)
        return [idx] if 0 <= idx < n_features else []

    if isinstance(show, np.ndarray):
        show = show.tolist()

    if isinstance(show, (list, tuple, set)):
        resolved = []

        for item in show:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if 0 <= idx < n_features:
                    resolved.append(idx)
                continue

            item_str = str(item)

            try:
                idx = int(item_str)
                if 0 <= idx < n_features:
                    resolved.append(idx)
            except ValueError:
                if item_str in cpd_ids:
                    resolved.append(cpd_ids.index(item_str))

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(resolved))

    raise ValueError(
        "show must be 'default', 'all', 'none', an int, a cpdID string, or a list."
    )


def _make_batch_colors(batch, cmap_name):
    plt, mpl, _ = _load_plotting()

    unique_batches = list(dict.fromkeys(batch))

    cmap = plt.get_cmap(cmap_name)

    if len(unique_batches) == 1:
        normalized_indices = {unique_batches[0]: 0.0}
    else:
        normalized_indices = {
            batch_id: index / max(1, len(unique_batches) - 1)
            for index, batch_id in enumerate(unique_batches)
        }

    batch_colors = [
        mpl.colors.rgb2hex(cmap(normalized_indices[batch_id]))
        for batch_id in unique_batches
    ]

    batch_to_color = {
        batch_id: batch_colors[i % len(batch_colors)]
        for i, batch_id in enumerate(unique_batches)
    }

    return unique_batches, batch_colors, batch_to_color


def _natural_sort_key(s):
    if isinstance(s, tuple):
        s = "-".join(map(str, s))

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def _safe_filename(value):
    value = str(value)
    value = re.sub(r'[<>:"/\\\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("_")
    return value or "value"


def _parse_metadata_columns(value):
    """
    Allows metadata grouping columns to come either as:
        "Diagnosis"
        "Diagnosis, Sex"
        "['Diagnosis', 'Sex']"
        ["Diagnosis", "Sex"]
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                raise ValueError(
                    "column_names must be a metadata column name or a list of names."
                )
            return parsed

        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

        return value

    return value


def _parse_show_value(value):
    """
    Converts GUI string inputs such as '[1, 2, 3]' or '1,2,3'
    before passing them to _resolve_feature_indices().
    """
    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"all", "default", "none"}:
            return value.lower()

        if value.startswith("[") and value.endswith("]"):
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                raise ValueError(
                    "indexes must be 'all', an integer, a cpdID, or a list."
                )
            return parsed

        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]

    return value


# ------------------------------------------------------------
# Visualization operations
# ------------------------------------------------------------


@register_operation(
    id="visualize_boxplot",
    label="Visualize Sample Boxplot",
    description=(
        "Create a boxplot of all sample intensity distributions. "
        "QC, blanks, dilution-series samples, and standards are color-coded."
    ),
    citation="",
    category_tags=[OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="names",
            type="bool",
            required=False,
            default=False,
            label="Show sample names",
            help="If True, show sample names on the x-axis. Can be cluttered for many samples.",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
            help="Suffix added to the saved plot filename.",
        ),
    ],
    requires=["data"],
    produces=["figures"],
)
def visualize_boxplot(
    state: WorkflowState,
    names: bool = False,
    plt_name_suffix: str = "",
):
    """
    Create a boxplot of all samples.

    This is useful for quick inspection of intensity distributions, possible
    batch effects, retention/alignment problems, or sample-level outliers.
    """
    plt, mpl, Line2D = _load_plotting()

    data = state.data

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if data.shape[1] <= 1:
        raise ValueError("Data must contain cpdID plus at least one sample column.")

    main_folder, figures_folder = _ensure_main_folders(state)
    suffixes = _get_suffixes(state)

    raw_data = data.iloc[:, 1:]
    numeric_data = raw_data.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    QC_set = _as_sample_set(getattr(state, "QC_samples", None))
    blank_set = _as_sample_set(getattr(state, "blank_samples", None))
    dilution_set = _as_sample_set(getattr(state, "dilution_series_samples", None))
    standard_set = _as_sample_set(getattr(state, "standard_samples", None))

    box_input = []
    kept_names = []

    for col_name in numeric_data.columns:
        values = numeric_data[col_name].dropna().values

        if values.size > 0:
            box_input.append(values)
            kept_names.append(col_name)

    if len(box_input) == 0:
        raise ValueError("No valid non-NaN values to plot. Check your input data.")

    is_qc_sample = [name in QC_set for name in kept_names]
    is_blank_sample = [name in blank_set for name in kept_names]
    is_dilution_series_sample = [name in dilution_set for name in kept_names]
    is_standard_sample = [name in standard_set for name in kept_names]

    fig, ax = plt.subplots(figsize=(18, 12))

    box = ax.boxplot(
        box_input,
        showfliers=False,
        showmeans=True,
        meanline=True,
        medianprops={"color": "black"},
        meanprops={"color": "blue"},
        patch_artist=True,
        whiskerprops={"color": "grey"},
        capprops={"color": "yellow"},
    )

    ax.set_title("Boxplot of all samples")
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Peak Area")

    colors = [
        (
            "grey"
            if qc
            else (
                "darkred"
                if blank
                else "blue" if dilution else "darkgreen" if standard else "lightblue"
            )
        )
        for qc, blank, dilution, standard in zip(
            is_qc_sample,
            is_blank_sample,
            is_dilution_series_sample,
            is_standard_sample,
        )
    ]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    n_samples = len(kept_names)

    if names:
        if n_samples > 20:
            ax.set_xticks(np.arange(1, n_samples + 1))
            ax.set_xticklabels(kept_names, rotation=90, fontsize=8)
        else:
            ax.set_xticks(np.arange(1, n_samples + 1))
            ax.set_xticklabels(kept_names, fontsize=10)

    elif n_samples > 50:
        step = max(1, n_samples // 10)
        tick_positions = np.arange(1, n_samples + 1, step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [kept_names[i - 1] for i in tick_positions],
            rotation=90,
            fontsize=8,
        )

    else:
        ax.set_xticks(np.arange(1, n_samples + 1))
        ax.set_xticklabels([""] * n_samples)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="QC samples",
            markerfacecolor="grey",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Blank samples",
            markerfacecolor="darkred",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Dilution series samples",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Standard samples",
            markerfacecolor="darkgreen",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Samples",
            markerfacecolor="lightblue",
            markersize=10,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=12,
        title="Sample types",
        title_fontsize="13",
        frameon=True,
    )

    plt_name = os.path.join(
        figures_folder,
        "QC_samples_boxplot_" + str(plt_name_suffix),
    )

    saved_paths = []

    for suffix in suffixes:
        out_path = plt_name + suffix
        out_path = _unique_path(out_path)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        saved_paths.append(out_path)

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Boxplot of sample intensity distributions was created for "
            f"{len(kept_names)} samples. "
            f"QC, blank, dilution-series, standard, and regular samples "
            f"are distinguished by color."
        ),
        title="Sample intensity boxplot",
    )

    add_figure(
        state,
        fig,
        title="Boxplot of all samples",
    )

    # Figure base path should reflect the actual saved filename (without extension)
    if len(saved_paths) > 0:
        first_saved = saved_paths[0]
        figure_base = os.path.splitext(first_saved)[0]
    else:
        figure_base = plt_name

    plt.close(fig)
    return {
        "figure_base_path": figure_base,
        "saved_paths": saved_paths,
        "n_samples_plotted": len(kept_names),
        "show_names": names,
    }


@register_operation(
    id="visualize_samples_by_batch",
    label="Visualize Samples by Batch",
    description=(
        "Visualize selected features across sample order, colored by batch and sample type. "
        "QC samples are highlighted and connected."
    ),
    citation="",
    category_tags=[OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="show",
            type="str_or_list",
            required=False,
            default="default",
            label="Features to show",
            help="'default', 'all', 'none', a row index, a cpdID, or a list of indexes/cpdIDs.",
        ),
        ParameterDef(
            name="cmap",
            type="str",
            required=False,
            default="viridis",
            label="Colormap",
            help="Matplotlib colormap used for batches.",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
            help="Suffix added to saved plot filenames.",
        ),
    ],
    requires=["data"],
    produces=["figures"],
)
def visualize_samples_by_batch(
    state: WorkflowState,
    show="default",
    cmap: str = "viridis",
    plt_name_suffix: str = "",
):
    """
    Visualize sample intensities for selected features.

    Samples are colored by:
        - QC
        - blank
        - dilution series
        - standard
        - batch
    """
    plt, mpl, Line2D = _load_plotting()

    data = state.data

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if data.shape[1] <= 1:
        raise ValueError("Data must contain cpdID plus at least one sample column.")

    main_folder, figures_folder = _ensure_main_folders(state)
    suffixes = _get_suffixes(state)

    sample_cols = list(data.columns[1:])
    n_samples = len(sample_cols)

    batch = getattr(state, "batch", None)

    if batch is None:
        batch = ["all_one_batch" for _ in range(n_samples)]
    else: 
        batch = list(batch)

    if len(batch) != n_samples:
        raise ValueError(
            f"Length of state.batch ({len(batch)}) does not match "
            f"number of sample columns ({n_samples})."
        )

    unique_batches, batch_colors, batch_to_color = _make_batch_colors(batch, cmap)

    QC_set = _as_sample_set(getattr(state, "QC_samples", None))
    blank_set = _as_sample_set(getattr(state, "blank_samples", None))
    dilution_set = _as_sample_set(getattr(state, "dilution_series_samples", None))
    standard_set = _as_sample_set(getattr(state, "standard_samples", None))

    is_qc_sample = np.array([col in QC_set for col in sample_cols], dtype=bool)
    is_blank_sample = np.array([col in blank_set for col in sample_cols], dtype=bool)
    is_dilution_series_sample = np.array(
        [col in dilution_set for col in sample_cols],
        dtype=bool,
    )
    is_standard_sample = np.array(
        [col in standard_set for col in sample_cols], dtype=bool
    )

    show = _parse_show_value(show)
    feature_indices = _resolve_feature_indices(show, data)

    saved_paths = []

    # REPORTING ---- part 1 ---------------------------------------------
    if len(feature_indices) == 0:
        add_text(
            state,
            "Samples-by-batch visualization was requested, but no valid features were selected.",
            title="Samples by batch",
        )
        return {
            "saved_paths": [],
            "features_plotted": [],
            "n_features_plotted": 0,
        }
    # ------------------------------------------------------------------

    for feature_idx in feature_indices:
        row_values = pd.to_numeric(
            data.iloc[feature_idx, 1:],
            errors="coerce",
        )

        row_array = row_values.to_numpy(dtype=float)

        point_colors = [
            (
                "black"
                if qc
                else (
                    "darkblue"
                    if blank
                    else (
                        "darkred"
                        if dilution
                        else "darkgreen" if standard else batch_to_color[batch_id]
                    )
                )
            )
            for batch_id, qc, blank, dilution, standard in zip(
                batch,
                is_qc_sample,
                is_blank_sample,
                is_dilution_series_sample,
                is_standard_sample,
            )
        ]

        alphas = [
            0.5 if qc else 0.1 if value == 0 else 0.8
            for qc, value in zip(is_qc_sample, row_array)
        ]

        zero_counts = {batch_id: 0 for batch_id in unique_batches}
        qc_zero_counts = {batch_id: 0 for batch_id in unique_batches}

        for batch_id, value, qc in zip(batch, row_array, is_qc_sample):
            if value == 0:
                zero_counts[batch_id] += 1

                if qc:
                    qc_zero_counts[batch_id] += 1

        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        fig = plt.figure(figsize=(20, 4))

        # Main scatter plot.
        ax = fig.add_subplot(gs[0])

        x = np.arange(n_samples)

        ax.scatter(
            x,
            row_array,
            color=point_colors,
            alpha=alphas,
            marker="o",
        )

        # Connect non-zero QC samples.
        qc_x = x[is_qc_sample]
        qc_y = row_array[is_qc_sample]

        qc_nonzero = np.isfinite(qc_y) & (qc_y != 0)

        if qc_nonzero.any():
            ax.plot(
                qc_x[qc_nonzero],
                qc_y[qc_nonzero],
                color="black",
                linewidth=1,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_samples)
        ax.set_xlabel("Samples in order")
        ax.set_ylabel("Peak Area")

        cpd_id = (
            str(data.iloc[feature_idx, 0]) if data.shape[1] > 0 else str(feature_idx)
        )

        ax.set_title("cpdID = " + cpd_id)

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="QC samples",
                markerfacecolor="black",
                markersize=10,
                alpha=0.5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Blank samples",
                markerfacecolor="darkblue",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Dilution series samples",
                markerfacecolor="darkred",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Standard samples",
                markerfacecolor="darkgreen",
                markersize=10,
            ),
        ]

        for batch_id, color in batch_to_color.items():
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Batch: " + str(batch_id),
                    markerfacecolor=color,
                    markersize=10,
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=10,
            bbox_to_anchor=(1, 1),
            title="Sample types",
            title_fontsize="13",
            frameon=False,
        )

        # Zero-count table.
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis("tight")
        ax_table.axis("off")
        fig.subplots_adjust(bottom=-0.5)

        total_samples = {batch_id: batch.count(batch_id) for batch_id in unique_batches}

        total_qc_samples = {
            batch_id: sum(
                1
                for batch_cur, qc in zip(batch, is_qc_sample)
                if batch_cur == batch_id and qc
            )
            for batch_id in unique_batches
        }

        zero_percentages = {
            batch_id: (
                zero_counts[batch_id] / total_samples[batch_id] * 100
                if total_samples[batch_id] > 0
                else 0
            )
            for batch_id in unique_batches
        }

        qc_zero_percentages = {
            batch_id: (
                qc_zero_counts[batch_id] / total_qc_samples[batch_id] * 100
                if total_qc_samples[batch_id] > 0
                else 0
            )
            for batch_id in unique_batches
        }

        formatted_zero_counts = [
            f"{zero_counts[batch_id]} ({zero_percentages[batch_id]:.2f}%)"
            for batch_id in unique_batches
        ]

        formatted_qc_zero_counts = [
            f"{qc_zero_counts[batch_id]} ({qc_zero_percentages[batch_id]:.2f}%)"
            for batch_id in unique_batches
        ]

        total_missing = sum(zero_counts.values())
        total_percentage = total_missing / len(batch) * 100 if len(batch) > 0 else 0

        total_qc_zero_count = sum(qc_zero_counts.values())
        total_qc_count = int(is_qc_sample.sum())
        total_qc_percentage = (
            total_qc_zero_count / total_qc_count * 100 if total_qc_count > 0 else 0
        )

        col_labels = unique_batches + ["Total"]

        formatted_zero_counts.append(f"{total_missing} ({total_percentage:.2f}%)")

        table_rows = [formatted_zero_counts]
        row_labels = ["All samples"]

        if total_qc_count > 0:
            formatted_qc_zero_counts.append(
                f"{total_qc_zero_count} ({total_qc_percentage:.2f}%)"
            )
            table_rows.append(formatted_qc_zero_counts)
            row_labels.append("QC samples")

        table_batch_colors = batch_colors + ["white"]
        table_batch_colors_rgba = [
            mpl.colors.to_rgba(color, alpha=0.6) for color in table_batch_colors
        ]

        cell_colours = [table_batch_colors_rgba for _ in table_rows]

        ax_table.table(
            cellText=table_rows,
            cellColours=cell_colours,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            fontsize=10,
            loc="center",
        )

        ax_table.text(
            x=-0.005,
            y=0.65,
            s="Zero counts",
            fontsize=15,
            transform=ax_table.transAxes,
            ha="right",
            va="center",
        )

        plt_name = os.path.join(
            figures_folder,
            "single_compound_view_" + str(feature_idx) + "_" + str(plt_name_suffix),
        )

        for suffix in suffixes:
            out_path = plt_name + suffix
            out_path = _unique_path(out_path)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            saved_paths.append(out_path)

    # REPORTING ---- part 2 ---------------------------------------------
    add_text(
        state,
        (
            f"Samples-by-batch visualization was created for "
            f"{len(feature_indices)} feature(s). "
            f"Samples were displayed in acquisition order and colored according "
            f"to batch and sample type. QC samples were highlighted and connected."
        ),
        title="Samples by batch",
    )
    add_figure(
        state,
        fig,
        title=f"Samples by batch — {cpd_id}",
    )
        
    plt.close(fig)
    return {
        "saved_paths": saved_paths,
        "features_plotted": feature_indices,
        "n_features_plotted": len(feature_indices),
        "cmap": cmap,
    }

@register_operation(
    id="visualizer_violin_plots",
    label="Violin Plots",
    description=(
        "Create violin plots for selected features grouped by one or more "
        "metadata columns and save them into a searchable HTML gallery."
    ),
    citation="",
    category_tags=[OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="column_names",
            type="str_or_list",
            required=True,
            default="",
            label="Grouping column(s)",
            help=(
                "Metadata column used for grouping, for example Diagnosis, "
                "Type, or ['Diagnosis', 'Sex']."
            ),
        ),
        ParameterDef(
            name="indexes",
            type="str_or_list",
            required=False,
            default="all",
            label="Feature indexes",
            help=(
                "Features to plot. Use 'all', 'default', one feature index, "
                "one cpdID, or a list of indexes/cpdIDs."
            ),
        ),
        ParameterDef(
            name="gallery_name",
            type="str",
            required=False,
            default="",
            label="Gallery name",
            help=(
                "Name of the output gallery. If empty, a name is generated "
                "from the grouping column(s). If the name already exists, "
                "a numeric suffix is added automatically."
            ),
        ),
        ParameterDef(
            name="cmap",
            type="str",
            required=False,
            default="nipy_spectral",
            label="Colormap",
            help="Matplotlib colormap used for groups.",
        ),
        ParameterDef(
            name="bw",
            type="float",
            required=False,
            default=0.2,
            label="Bandwidth",
            help="Bandwidth used for violin kernel-density estimation.",
        ),
        ParameterDef(
            name="jitter",
            type="bool",
            required=False,
            default=True,
            label="Jitter points",
            help="If True, individual observations are horizontally jittered.",
        ),
        ParameterDef(
            name="label_rotation",
            type="int",
            required=False,
            default=0,
            label="Label rotation",
            help="Rotation of x-axis group labels.",
        ),
    ],
    requires=["data", "metadata"],
    produces=["figures"],
)
def visualizer_violin_plots(
    state: WorkflowState,
    column_names="",
    indexes="all",
    gallery_name="",
    cmap="nipy_spectral",
    bw=0.2,
    jitter=True,
    label_rotation=0,
):
    """
    Create violin plots for selected features.

    Every successfully generated feature plot is stored as an individual PNG
    inside a dedicated gallery directory. A searchable HTML gallery and a CSV
    index are generated so plots can later be located by cpdID or feature index.

    Only one example figure is embedded in the main PySPRESSO workflow report
    to prevent report.html from becoming excessively large.
    """

    plt, mpl, _ = _load_plotting()

    # Validate inputs
    if state.data is None:
        raise ValueError(
            "No data found. Run dataset initialization first."
        )
    if state.metadata is None:
        raise ValueError(
            "No metadata found. Run dataset initialization first."
        )
    if column_names is None or str(column_names).strip() == "":
        raise ValueError(
            "No grouping metadata column was provided. "
            "Set column_names, for example 'Diagnosis', "
            "'Type', or ['Diagnosis', 'Sex']."
        )
    try:
        bw = float(bw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Violin bandwidth must be numeric."
        ) from exc

    if not np.isfinite(bw) or bw <= 0:
        raise ValueError(
            "Violin bandwidth must be greater than zero."
        )

    try:
        label_rotation = int(label_rotation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "label_rotation must be an integer."
        ) from exc

    data = state.data.copy()
    metadata = state.metadata.copy()

    _, figures_folder = _ensure_main_folders(state)

    
    # Local helpers
    def _is_generated_extra_column(col):
        text = str(col).strip()
        lowered = text.lower()

        return (
            text == ""
            or lowered.startswith("extra_")
            or lowered.startswith("unnamed:")
            or "additional item" in lowered
            or "additional items" in lowered
        )

    def _extract_study_id_from_sample_column(sample_name):
        text = str(sample_name)

        matches = re.findall(
            r"\(([^()]*)\)",
            text,
        )

        if matches:
            return matches[-1]

        return text

    def _align_metadata_to_sample_columns(
        metadata_df,
        sample_columns,
    ):
        """
        Align metadata rows to current data sample columns.

        Sample File is preferred. Other common identifier columns are used
        as fallbacks. Positional matching is used only when row and sample
        counts match exactly.
        """

        sample_columns = list(sample_columns)

        meta = (
            metadata_df
            .copy()
            .reset_index(drop=True)
        )

        # Best case: exact Sample File matching
        if "Sample File" in meta.columns:
            temp = meta.copy()

            temp["__sample_key__"] = (
                temp["Sample File"]
                .astype(str)
                .str.strip()
            )
            duplicate_keys = temp.loc[
                temp["__sample_key__"].duplicated(
                    keep=False
                ),
                "__sample_key__",
            ].unique()

            if len(duplicate_keys):
                raise ValueError(
                    "Metadata contains duplicate "
                    "'Sample File' values: "
                    + ", ".join(
                        map(
                            str,
                            duplicate_keys[:10],
                        )
                    )
                )
            temp = temp.set_index(
                "__sample_key__"
            )

            keys = [
                str(col).strip()
                for col in sample_columns
            ]
            if all(
                key in temp.index
                for key in keys
            ):
                return (
                    temp
                    .loc[keys]
                    .reset_index(drop=True)
                )

        # Fallback matching
        candidate_cols = [
            "Study File ID",
            "Sample File",
            "Sample Name",
            "Sample ID",
            "File Name",
            "Name",
        ]

        for candidate_col in candidate_cols:

            if candidate_col not in meta.columns:
                continue

            values = (
                meta[candidate_col]
                .astype(str)
                .str.strip()
            )

            used = set()
            rows = []

            for sample_col in sample_columns:

                sample_text = (
                    str(sample_col)
                    .strip()
                )

                study_id = (
                    _extract_study_id_from_sample_column(
                        sample_text
                    )
                )

                match_idx = None

                for idx, value in values.items():

                    if idx in used:
                        continue

                    if (
                        value == sample_text
                        or value == study_id
                    ):
                        match_idx = idx
                        break

                if match_idx is None:
                    rows = []
                    break

                used.add(match_idx)

                rows.append(
                    meta.loc[match_idx]
                )

            if len(rows) == len(sample_columns):
                return (
                    pd.DataFrame(rows)
                    .reset_index(drop=True)
                )

        # Positional fallback only when unambiguous
        if len(meta) == len(sample_columns):
            return meta.reset_index(
                drop=True
            )

        raise ValueError(
            "Metadata could not be aligned to the current "
            "data sample columns. "
            f"Metadata rows: {len(meta)}; "
            f"sample columns: {len(sample_columns)}. "
            "Expected metadata['Sample File'] or another "
            "sample identifier to match the data columns."
        )

    def _unique_directory(path):
        """
        Return a new directory name without overwriting an existing gallery.
        """

        if not os.path.exists(path):
            return path

        i = 1

        while True:
            candidate = f"{path}_{i}"

            if not os.path.exists(candidate):
                return candidate

            i += 1

    def _write_gallery(
        gallery_path,
        plot_items,
        grouping_label,
        gallery_title,
        skipped_count,
    ):
        """
        Create a lightweight searchable HTML gallery.

        Images remain separate PNG files and are lazy-loaded by the browser.
        """

        from html import escape

        cards = []

        for item in plot_items:

            image_name = item["image"]
            cpd_id = str(item["cpdID"])
            feature_index = int(
                item["feature_index"]
            )

            search_text = (
                f"{feature_index} {cpd_id}"
                .lower()
            )

            cards.append(
                f"""
                <article
                    class="violin-card"
                    data-search="{escape(search_text, quote=True)}"
                >
                    <div class="violin-caption">
                        <strong>{escape(cpd_id)}</strong>
                        <span>
                            feature index: {feature_index}
                        </span>
                    </div>

                    <a
                        href="{escape(image_name, quote=True)}"
                        target="_blank"
                    >
                        <img
                            src="{escape(image_name, quote=True)}"
                            loading="lazy"
                            decoding="async"
                            alt="{escape(cpd_id, quote=True)}"
                        >
                    </a>
                </article>
                """
            )

        html = f"""<!doctype html>
<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>{escape(gallery_title)}</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 24px;

    background: #fafafa;
}}

.gallery-header {{
    position: sticky;

    top: 0;

    z-index: 10;

    background: #fafafa;

    padding:
        10px
        0
        18px
        0;

    border-bottom:
        1px solid
        #ddd;
}}

.gallery-header h1 {{
    margin-bottom: 8px;
}}

.gallery-meta {{
    margin:
        4px
        0
        14px
        0;

    color: #555;
}}

#violin-search {{
    width:
        min(
            600px,
            95%
        );

    box-sizing:
        border-box;

    padding: 10px;

    font-size: 16px;
}}

.violin-grid {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(
                340px,
                1fr
            )
        );

    gap: 20px;

    margin-top: 20px;
}}

.violin-card {{
    background: white;

    border:
        1px solid
        #ddd;

    border-radius: 8px;

    padding: 10px;
}}

.violin-card img {{
    display: block;

    width: 100%;

    height: auto;
}}

.violin-caption {{
    display: flex;

    justify-content:
        space-between;

    gap: 10px;

    margin-bottom: 8px;
}}

.violin-caption span {{
    color: #666;

    font-size: 0.9em;
}}

</style>

</head>

<body>

<div class="gallery-header">

    <h1>
        {escape(gallery_title)}
    </h1>

    <p class="gallery-meta">
        Grouping:
        {escape(str(grouping_label))}
        &nbsp;|&nbsp;
        Plots:
        {len(plot_items)}
        &nbsp;|&nbsp;
        Skipped:
        {int(skipped_count)}
    </p>

    <input
        id="violin-search"
        type="search"
        placeholder="Search by cpdID or feature index..."
    >

</div>

<div class="violin-grid">

    {''.join(cards)}

</div>

<script>

const search =
    document.getElementById(
        "violin-search"
    );

const cards =
    document.querySelectorAll(
        ".violin-card"
    );

search.addEventListener(
    "input",
    () => {{

        const query =
            search
                .value
                .trim()
                .toLowerCase();

        cards.forEach(
            card => {{

                card.hidden =
                    query.length > 0
                    &&
                    !card.dataset.search.includes(
                        query
                    );

            }}
        );

    }}
);

</script>

</body>

</html>
"""
        with open(
            gallery_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(html)

    
    # Clean and identify current sample columns
    if "cpdID" not in data.columns:
        raise ValueError(
            "Expected 'cpdID' column in state.data."
        )

    sample_columns = [
        col
        for col in data.columns
        if (
            col != "cpdID"
            and not _is_generated_extra_column(
                col
            )
        )
    ]

    removed_generated_columns = [
        col
        for col in data.columns
        if (
            col != "cpdID"
            and _is_generated_extra_column(
                col
            )
        )
    ]
    if removed_generated_columns:
        print(
            "[violin] Ignoring generated/non-sample "
            "columns: "
            + str(
                [
                    str(c)
                    for c
                    in removed_generated_columns
                ]
            )
        )

    if len(data) == 0:
        raise ValueError(
            "Data table is empty; no violin "
            "plots can be created."
        )
    if len(sample_columns) == 0:
        raise ValueError(
            "No sample columns found in state.data."
        )

    data = data[
        ["cpdID"] + sample_columns
    ].copy()

    aligned_metadata = (
        _align_metadata_to_sample_columns(
            metadata,
            sample_columns,
        )
    )

    # Resolve metadata grouping
    parsed_columns = (
        _parse_metadata_columns(
            column_names
        )
    )
    if isinstance(
        parsed_columns,
        (list, tuple),
    ):
        grouping_columns = [
            str(col).strip()
            for col in parsed_columns
            if str(col).strip()
        ]
    else:
        grouping_columns = [
            str(parsed_columns).strip()
        ]
    if not grouping_columns:
        raise ValueError(
            "No valid grouping metadata "
            "columns were provided."
        )

    missing_grouping_columns = [
        col
        for col in grouping_columns
        if col not in aligned_metadata.columns
    ]
    if missing_grouping_columns:
        raise ValueError(
            "Metadata grouping column(s) "
            "were not found: "
            + ", ".join(
                missing_grouping_columns
            )
        )
    grouping_label_for_axis = (
        ", ".join(
            grouping_columns
        )
    )
    grouping_frame = (
        aligned_metadata[
            grouping_columns
        ]
        .copy()
    )

    for col in grouping_columns:
        grouping_frame[col] = (
            grouping_frame[col]
            .astype(object)
            .where(
                grouping_frame[col]
                .notna(),
                "None",
            )
            .astype(str)
        )

    if len(grouping_columns) == 1:
        grouping_values = (
            grouping_frame[
                grouping_columns[0]
            ]
            .to_numpy()
        )

    else:
        grouping_values = (
            grouping_frame
            .agg(
                " | ".join,
                axis=1,
            )
            .to_numpy()
        )

    # Resolve requested features
    indexes = _parse_show_value(
        indexes
    )
    resolved_indexes = (
        _resolve_feature_indices(
            indexes,
            data,
        )
    )
    if not resolved_indexes:
        raise ValueError(
            "No valid feature indexes were "
            "resolved from the indexes parameter."
        )
    resolved_indexes = list(
        dict.fromkeys(
            resolved_indexes
        )
    )
    bad_indexes = [
        idx
        for idx in resolved_indexes
        if (
            idx < 0
            or idx >= len(data)
        )
    ]
    if bad_indexes:
        raise ValueError(
            "Some feature indexes are "
            f"out of range: {bad_indexes}"
        )

    # Prepare numeric matrix once
    numeric_matrix = (
        data[sample_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .T
        .to_numpy(
            dtype=float
        )
    )

    feature_ids = (
        data["cpdID"]
        .astype(str)
        .tolist()
    )

    
    # Prepare grouping indexes
    group_order = sorted(
        pd.unique(
            grouping_values
        ).tolist(),
        key=_natural_sort_key,
    )
    if len(group_order) == 0:
        raise ValueError(
            "No non-empty metadata groups "
            "were found."
        )
    group_indices = {
        group:
            np.flatnonzero(
                grouping_values
                == group
            )
        for group in group_order
    }

    # Move QC-containing group to the end
    QC_samples = getattr(
        state,
        "QC_samples",
        None,
    )
    qc_key = None

    if QC_samples:
        qc_set = set(
            map(
                str,
                QC_samples,
            )
        )
        best_key = None
        best_overlap = 0
        for group in group_order:

            sample_indexes = (
                group_indices[group]
            )
            group_sample_names = {
                str(
                    sample_columns[i]
                )
                for i
                in sample_indexes
            }
            overlap = len(
                group_sample_names
                & qc_set
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = group

        if best_overlap > 0:
            qc_key = best_key

    if (
        qc_key is not None
        and qc_key in group_order
    ):
        group_order.append(
            group_order.pop(
                group_order.index(
                    qc_key
                )
            )
        )

    # Group colors and labels
    try:
        cmap_obj = (
            mpl.colormaps
            .get_cmap(cmap)
        )
    except ValueError as exc:
        raise ValueError(
            f"Unknown matplotlib colormap: {cmap}"
        ) from exc

    if len(group_order) == 1:
        color_indices = [0.5]
    else:
        color_indices = (
            np.linspace(
                0.05,
                0.95,
                len(group_order),
            )
        )
    group_colors = {
        group:
            cmap_obj(
                color_indices[i]
            )
        for i, group
        in enumerate(
            group_order
        )
    }
    x_labels = [
        (
            f"{group} "
            f"({len(group_indices[group])})"
        )
        for group in group_order
    ]

    # Create uniquely named gallery
    requested_gallery_name = (
        str(
            gallery_name
            or ""
        )
        .strip()
    )

    if requested_gallery_name:
        gallery_stem = (
            _safe_filename(
                requested_gallery_name
            )
        )
    else:
        safe_grouping_name = (
            _safe_filename(
                grouping_label_for_axis
            )
        )
        gallery_stem = (
            f"violin_{safe_grouping_name}"
        )

    gallery_dir = (
        _unique_directory(
            os.path.join(
                figures_folder,
                gallery_stem,
            )
        )
    )

    os.makedirs(
        gallery_dir,
        exist_ok=False,
    )

    actual_gallery_name = (
        os.path.basename(
            gallery_dir
        )
    )

    gallery_path = os.path.join(
        gallery_dir,
        f"{actual_gallery_name}.html",
    )

    index_path = os.path.join(
        gallery_dir,
        f"{actual_gallery_name}_index.csv",
    )

    # Plot helper
    def _create_violin_figure(
        feature_index,
    ):
        cpd_title = feature_ids[
            feature_index
        ]
        values_by_group = []
        for (
            position,
            group_name,
        ) in enumerate(
            group_order,
            start=1,
        ):
            sample_indexes = (
                group_indices[
                    group_name
                ]
            )
            values = (
                numeric_matrix[
                    sample_indexes,
                    feature_index,
                ]
            )
            values = values[
                np.isfinite(
                    values
                )
            ]

            if values.size == 0:
                continue
            values_by_group.append(
                {
                    "position":
                        position,
                    "group":
                        group_name,
                    "values":
                        values,
                    "color":
                        group_colors[
                            group_name
                        ],
                }
            )
        if not values_by_group:
            return None, cpd_title

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )
        rng = (
            np.random
            .default_rng(
                42
                + int(
                    feature_index
                )
            )
        )

        # Draw all meaningful violins in one matplotlib call
        violin_entries = [
            entry
            for entry
            in values_by_group
            if (
                len(
                    entry["values"]
                ) >= 2
                and not np.allclose(
                    entry["values"],
                    entry["values"][0],
                )
            )
        ]
        if violin_entries:
            violin = (
                ax.violinplot(
                    [
                        entry["values"]
                        for entry
                        in violin_entries
                    ],
                    positions=[
                        entry["position"]
                        for entry
                        in violin_entries
                    ],
                    widths=0.8,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    bw_method=bw,
                )
            )
            for (
                body,
                entry,
            ) in zip(
                violin["bodies"],
                violin_entries,
            ):
                body.set_facecolor(
                    entry["color"]
                )
                body.set_edgecolor(
                    entry["color"]
                )
                body.set_linewidth(
                    1
                )
                body.set_alpha(
                    0.75
                )

        # Means, medians and observations
        for entry in values_by_group:
            position = (
                entry["position"]
            )
            values = (
                entry["values"]
            )
            color = (
                entry["color"]
            )
            mean_value = float(
                np.mean(values)
            )
            median_value = float(
                np.median(values)
            )
            ax.plot(
                [
                    position - 0.20,
                    position + 0.20,
                ],
                [
                    mean_value,
                    mean_value,
                ],
                color=color,
                linewidth=0.7,
            )
            ax.plot(
                [
                    position - 0.12,
                    position + 0.12,
                ],
                [
                    median_value,
                    median_value,
                ],
                color=color,
                linewidth=1.5,
            )
            if (
                jitter
                and len(values) > 1
            ):
                x_values = (
                    position
                    + rng.uniform(
                        -0.06,
                        0.06,
                        size=len(values),
                    )
                )

            else:
                x_values = (
                    np.full(
                        len(values),
                        position,
                    )
                )

            ax.scatter(
                x_values,
                values,
                color=color,
                s=6,
                alpha=0.9,
            )
        ax.set_xticks(
            np.arange(
                1,
                len(group_order) + 1,
            )
        )
        ax.set_xticklabels(
            x_labels,
            rotation=label_rotation,
        )
        ax.set_title(
            str(cpd_title)
        )
        ax.set_xlabel(
            grouping_label_for_axis
        )
        ax.set_ylabel(
            "Intensity"
        )

        fig.tight_layout()

        return fig, cpd_title


    # Create all requested plots
    plotted_features = []
    skipped_features = []

    example_path = None
    for (
        idx_position,
        feature_index,
    ) in enumerate(
        resolved_indexes
    ):
        fig, cpd_title = (
            _create_violin_figure(
                feature_index
            )
        )
        if fig is None:
            skipped_features.append(
                {
                    "feature_index":
                        int(
                            feature_index
                        ),
                    "cpdID":
                        str(
                            cpd_title
                        ),
                    "reason":
                        (
                            "No finite values "
                            "available in any group"
                        ),
                }
            )
            continue
        safe_cpd = (
            _safe_filename(
                str(
                    cpd_title
                )
            )[:100]
        )
        png_name = (
            f"{int(feature_index):06d}_"
            f"{safe_cpd}.png"
        )
        png_path = (
            os.path.join(
                gallery_dir,
                png_name,
            )
        )
        try:
            fig.savefig(
                png_path,
                dpi=140,
            )
        finally:

            plt.close(fig)
        if example_path is None:
            example_path = (
                png_path
            )
        plotted_features.append(
            {
                "feature_index":
                    int(
                        feature_index
                    ),
                "cpdID":
                    str(
                        cpd_title
                    ),
                "image":
                    png_name,
                "image_path":
                    png_path,
            }
        )
        print(
            (
                "Violin plots created: "
                f"{((idx_position + 1) / len(resolved_indexes)) * 100:.2f}%"
            ),
            end=(
                "\n"
                if (
                    idx_position
                    == len(
                        resolved_indexes
                    ) - 1
                )
                else "\r"
            ),
        )

    # Ensure something was actually generated
    if not plotted_features:
        try:
            os.rmdir(
                gallery_dir
            )
        except OSError:
            pass
        raise ValueError(
            "No violin plots could be created because "
            "none of the selected features contained "
            "finite values in any group."
        )

    # Create CSV index
    index_rows = []
    for item in plotted_features:
        index_rows.append(
            {
                "feature_index":
                    item[
                        "feature_index"
                    ],
                "cpdID":
                    item[
                        "cpdID"
                    ],
                "status":
                    "plotted",
                "image":
                    item[
                        "image"
                    ],
                "reason":
                    "",
            }
        )

    for item in skipped_features:
        index_rows.append(
            {
                "feature_index":
                    item[
                        "feature_index"
                    ],
                "cpdID":
                    item[
                        "cpdID"
                    ],
                "status":
                    "skipped",
                "image":
                    "",
                "reason":
                    item[
                        "reason"
                    ],
            }
        )
    index_table = (
        pd.DataFrame(
            index_rows
        )
        .sort_values(
            "feature_index"
        )
        .reset_index(
            drop=True
        )
    )
    index_table.to_csv(
        index_path,
        index=False,
        sep=";",
    )

    # Create searchable HTML gallery
    _write_gallery(
        gallery_path=gallery_path,
        plot_items=plotted_features,
        grouping_label=grouping_label_for_axis,
        gallery_title=actual_gallery_name,
        skipped_count=len(
            skipped_features
        ),
    )

    # Register compact artifacts only
    artifacts = getattr(
        state,
        "artifacts",
        None,
    )

    if artifacts is not None:
        artifacts.append(
            {
                "type":
                    "html",
                "path":
                    gallery_path,
                "description":
                    (
                        "Searchable violin "
                        "plot gallery"
                    ),
            }
        )
        artifacts.append(
            {
                "type":
                    "table",
                "path":
                    index_path,
                "description":
                    "Violin plot index",
            }
        )

        if example_path is not None:
            artifacts.append(
                {
                    "type":
                        "figure",
                    "path":
                        example_path,
                    "description":
                        (
                            "Example violin "
                            "plot"
                        ),
                }
            )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Violin plots were created for "
            f"{len(plotted_features)} of "
            f"{len(resolved_indexes)} requested features. "
            f"Features were grouped by "
            f"'{grouping_label_for_axis}'. "
            f"The plots were saved into the searchable "
            f"gallery '{actual_gallery_name}'."
        ),
        title="Violin plots",
    )

    add_text(
        state,
        f"Gallery: {gallery_path}",
        title="Violin plot gallery",
    )

    if example_path is not None:

        add_figure(
            state,
            example_path,
            title="Example violin plot",
        )

    if skipped_features:

        add_table(
            state,
            pd.DataFrame(
                skipped_features
            ),
            title="Skipped violin plots",
            include_index=False,
            max_rows=50,
        )

    # ------------------------------------------------------------------
    # Return compact operation result
    # ------------------------------------------------------------------

    return {
        "gallery_name":
            actual_gallery_name,
        "gallery_path":
            gallery_path,
        "gallery_directory":
            gallery_dir,
        "index_path":
            index_path,
        "n_requested":
            len(
                resolved_indexes
            ),
        "n_plots":
            len(
                plotted_features
            ),
        "n_skipped":
            len(
                skipped_features
            ),
        "example_path":
            example_path,
    }
