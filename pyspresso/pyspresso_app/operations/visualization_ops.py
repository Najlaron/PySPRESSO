# pyspresso_app/operations/visualization_ops.py

import os

import numpy as np
import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState


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
            return np.linspace(0, n_features - 1, min(5, n_features), dtype=int).tolist()

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

    cmap = mpl.cm.get_cmap(cmap_name)

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

    report = state.report

    main_folder, figures_folder = _ensure_main_folders(state)
    suffixes = _get_suffixes(state)

    raw_data = data.iloc[:, 1:]
    numeric_data = (
        raw_data
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
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
            else "darkred"
            if blank
            else "blue"
            if dilution
            else "darkgreen"
            if standard
            else "lightblue"
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
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        saved_paths.append(out_path)

    plt.close(fig)

    if report is not None:
        text0 = "The initial visualization of the data was created."
        text1 = "The boxplot of all samples was created and saved to: " + plt_name

        report.add_together(
            [
                ("text", text0),
                ("text", text1),
                ("image", plt_name + ".png"),
                "line",
            ]
        )

    return {
        "figure_base_path": plt_name,
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

    report = state.report

    main_folder, figures_folder = _ensure_main_folders(state)
    suffixes = _get_suffixes(state)

    sample_cols = list(data.columns[1:])
    n_samples = len(sample_cols)

    batch = getattr(state, "batch", None)

    if batch is None:
        batch = ["all_one_batch" for _ in range(n_samples)]
        state.batch = batch

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
    is_standard_sample = np.array([col in standard_set for col in sample_cols], dtype=bool)

    feature_indices = _resolve_feature_indices(show, data)

    saved_paths = []

    if len(feature_indices) == 0:
        if report is not None:
            report.add_together(
                [
                    (
                        "text",
                        "Samples-by-batch visualization was requested, but no features were selected.",
                    ),
                    "line",
                ]
            )

        return {
            "saved_paths": [],
            "features_plotted": [],
            "n_features_plotted": 0,
        }

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
                else "darkblue"
                if blank
                else "darkred"
                if dilution
                else "darkgreen"
                if standard
                else batch_to_color[batch_id]
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
            str(data.iloc[feature_idx, 0])
            if data.shape[1] > 0
            else str(feature_idx)
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

        total_samples = {
            batch_id: batch.count(batch_id)
            for batch_id in unique_batches
        }

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
            total_qc_zero_count / total_qc_count * 100
            if total_qc_count > 0
            else 0
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
            mpl.colors.to_rgba(color, alpha=0.6)
            for color in table_batch_colors
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
            "single_compound_view_"
            + str(feature_idx)
            + "_"
            + str(plt_name_suffix),
        )

        for suffix in suffixes:
            out_path = plt_name + suffix
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            saved_paths.append(out_path)

        plt.close(fig)

    if report is not None:
        text = "View of samples for selected compounds with highlighted QC samples was created."
        report.add_text(text)

        for image in saved_paths:
            if image.endswith(".png"):
                report.add_image(image)

        report.add_pagebreak()

    return {
        "saved_paths": saved_paths,
        "features_plotted": feature_indices,
        "n_features_plotted": len(feature_indices),
        "cmap": cmap,
    }