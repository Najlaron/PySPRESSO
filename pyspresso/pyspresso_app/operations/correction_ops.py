# pyspresso_app/operations/correction_ops.py

import os
import re
import time
import warnings

import numpy as np
import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState

from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)

# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def _filter_match_variable_metadata(data, variable_metadata):
    """
    Keep only variable metadata rows whose cpdID still exists in data.
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


def _ensure_main_folders(state: WorkflowState):
    """
    Ensure state.main_folder and figures folder exist.
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


def _resolve_show_indices(show, n_features, feature_names):
    """
    Resolve show parameter into a set of feature indexes.

    Supported:
        "default" -> 5 evenly spaced features
        "all"     -> all features
        "none"    -> no plots
        int       -> one feature index
        list      -> feature indexes or feature names/cpdIDs
        str       -> feature index or feature name/cpdID
    """
    if n_features == 0:
        return set()

    if show is None:
        return set()

    if isinstance(show, str):
        if show == "default":
            return set(np.linspace(0, n_features - 1, min(5, n_features), dtype=int))

        if show == "all":
            return set(range(n_features))

        if show == "none":
            return set()

        try:
            idx = int(show)
            return {idx} if 0 <= idx < n_features else set()
        except ValueError:
            feature_names_str = [str(x) for x in feature_names]
            if show in feature_names_str:
                return {feature_names_str.index(show)}
            return set()

    if isinstance(show, (int, np.integer)):
        idx = int(show)
        return {idx} if 0 <= idx < n_features else set()

    if isinstance(show, np.ndarray):
        show = show.tolist()

    if isinstance(show, (list, tuple, set)):
        resolved = set()
        feature_names_str = [str(x) for x in feature_names]

        for x in show:
            if isinstance(x, (int, np.integer)):
                idx = int(x)
                if 0 <= idx < n_features:
                    resolved.add(idx)
            else:
                xs = str(x)
                try:
                    idx = int(xs)
                    if 0 <= idx < n_features:
                        resolved.add(idx)
                except ValueError:
                    if xs in feature_names_str:
                        resolved.add(feature_names_str.index(xs))

        return resolved

    raise ValueError(
        "show must be 'default', 'all', 'none', an int, a string, or a list."
    )



def _sample_id_matches_column_name(sample_column, sample_id):
    """
    Match a batch-info sample identifier to a current data sample column.

    Compound Discoverer area columns often look like:
        Area: sample_name.raw (F25)

    while batch_info may contain only:
        F25
    """
    sample_column = str(sample_column)
    sample_id = str(sample_id)

    if sample_column == sample_id:
        return True

    if re.search(rf"\({re.escape(sample_id)}\)", sample_column):
        return True

    return False


def _is_generated_extra_column_name(column_name):
    """
    Detect columns generated by malformed table restoration / UI overflow.

    These are not real LC-MS sample columns and must never be used for correction.
    """
    name = str(column_name).strip()
    lower = name.lower()

    if lower.startswith("extra_"):
        return True

    if "additional" in lower and (name.startswith("<") or "item" in lower):
        return True

    if lower in {"additional_items", "additional items", "__extra__"}:
        return True

    return False


def _clean_qc_correction_data_columns(state: WorkflowState, df_in: pd.DataFrame):
    """
    Return data with only real sample columns for QC correction.

    The current bug was caused by an extra/generated column being present in
    state.data after state restoration/display. The original correction expected
    every column after cpdID to be a real sample column. This helper preserves the
    original correction logic, but removes generated non-sample columns before the
    samples x features matrix is created.
    """
    if "cpdID" not in df_in.columns:
        raise ValueError("Expected 'cpdID' column in state.data.")

    sample_columns = list(df_in.columns[1:])
    dropped_columns = []

    # First, remove obvious generated columns such as extra_0 or
    # '<1 additional items...'.
    generated_columns = [
        col for col in sample_columns if _is_generated_extra_column_name(col)
    ]

    if generated_columns:
        dropped_columns.extend(generated_columns)
        sample_columns = [col for col in sample_columns if col not in generated_columns]

    # If metadata has the authoritative current sample list, use it to remove
    # any other accidental non-sample columns. We only do this when most current
    # data columns match metadata, so we do not accidentally delete valid data in
    # unusual partially initialized workflows.
    metadata = getattr(state, "metadata", None)

    if isinstance(metadata, pd.DataFrame) and "Sample File" in metadata.columns:
        metadata_samples = [
            str(x) for x in metadata["Sample File"].dropna().tolist()
        ]
        metadata_sample_set = set(metadata_samples)

        matching_columns = [col for col in sample_columns if str(col) in metadata_sample_set]

        if matching_columns:
            # Typical case: one generated extra column exists. More generally,
            # require that metadata explains almost all current columns.
            allowed_mismatch = max(2, int(0.05 * max(1, len(sample_columns))))

            if len(sample_columns) - len(matching_columns) <= allowed_mismatch:
                metadata_dropped = [
                    col for col in sample_columns if str(col) not in metadata_sample_set
                ]
                dropped_columns.extend(metadata_dropped)
                sample_columns = matching_columns

    if len(sample_columns) == 0:
        raise ValueError(
            "No real sample columns remained after removing generated/non-sample columns."
        )

    if dropped_columns:
        unique_dropped = list(dict.fromkeys([str(c) for c in dropped_columns]))
        print(
            "[qc-correction] Ignoring non-sample/generated data columns: "
            + str(unique_dropped)
        )
        df_in = df_in[["cpdID"] + sample_columns].copy()
        state.data = df_in

    return df_in, sample_columns, list(dict.fromkeys([str(c) for c in dropped_columns]))


def _align_batch_to_sample_names(state: WorkflowState, sample_names):
    """
    Return batch labels aligned exactly to the current real sample columns.

    The original correction failed when state.batch still contained labels for a
    stale/generated column. This keeps original behavior when lengths already
    match, but repairs common cases using metadata/batch_info.
    """
    sample_names = [str(sample) for sample in sample_names]
    batch = getattr(state, "batch", None)

    if batch is None:
        aligned_batch = ["all_one_batch"] * len(sample_names)
        state.batch = aligned_batch
        return aligned_batch, "state.batch was missing; all samples were treated as one batch."

    batch = list(batch)

    if len(batch) == len(sample_names):
        return batch, ""

    # Safe one-batch repair: if all stored labels are identical, exact row
    # alignment is irrelevant.
    non_empty_labels = [
        str(x)
        for x in batch
        if x is not None and str(x).strip() != "" and str(x).lower() != "nan"
    ]
    unique_labels = list(dict.fromkeys(non_empty_labels))

    if len(unique_labels) == 1:
        aligned_batch = [unique_labels[0]] * len(sample_names)
        state.batch = aligned_batch
        note = (
            "state.batch length did not match current sample columns "
            f"({len(batch)} vs {len(sample_names)}), but all stored batch labels were "
            f"'{unique_labels[0]}'. The label was repeated for all current samples."
        )
        print("[batch-align] " + note)
        return aligned_batch, note

    def _try_align_from_table(table, table_name):
        if not isinstance(table, pd.DataFrame):
            return None, None

        if "Batch" not in table.columns:
            return None, None

        table = table.copy()

        if "Sample File" in table.columns:
            tmp = table[["Sample File", "Batch"]].dropna(subset=["Sample File"]).copy()
            tmp["Sample File"] = tmp["Sample File"].astype(str)
            tmp["Batch"] = tmp["Batch"].astype(str)
            tmp = tmp.drop_duplicates(subset=["Sample File"], keep="first")

            batch_map = dict(zip(tmp["Sample File"], tmp["Batch"]))
            missing = [sample for sample in sample_names if sample not in batch_map]

            if not missing:
                return (
                    [batch_map[sample] for sample in sample_names],
                    f"Batch labels were realigned from {table_name}['Sample File'] and {table_name}['Batch'].",
                )

        candidate_id_columns = [
            "Study File ID",
            "File ID",
            "Sample ID",
            "Sample_ID",
            "Sample Name",
            "File Name",
        ]

        for id_col in candidate_id_columns:
            if id_col not in table.columns:
                continue

            rows = table[[id_col, "Batch"]].dropna(subset=[id_col]).copy()
            rows[id_col] = rows[id_col].astype(str)
            rows["Batch"] = rows["Batch"].astype(str)

            aligned = []
            missing = []

            for sample_name in sample_names:
                matches = rows[
                    rows[id_col].apply(
                        lambda sample_id: _sample_id_matches_column_name(sample_name, sample_id)
                    )
                ]

                if len(matches) == 0:
                    aligned.append(None)
                    missing.append(sample_name)
                else:
                    aligned.append(str(matches.iloc[0]["Batch"]))

            if not missing:
                return (
                    aligned,
                    f"Batch labels were realigned from {table_name}['{id_col}'] matched to current data columns.",
                )

        if len(table) == len(sample_names):
            return (
                table["Batch"].astype(str).tolist(),
                f"Batch labels were realigned from {table_name}['Batch'] by row order.",
            )

        return None, None

    aligned_batch, align_note = _try_align_from_table(getattr(state, "metadata", None), "metadata")
    if aligned_batch is not None:
        state.batch = aligned_batch
        note = (
            "state.batch length did not match current sample columns "
            f"({len(batch)} vs {len(sample_names)}). {align_note}"
        )
        print("[batch-align] " + note)
        return aligned_batch, note

    aligned_batch, align_note = _try_align_from_table(getattr(state, "batch_info", None), "batch_info")
    if aligned_batch is not None:
        state.batch = aligned_batch
        note = (
            "state.batch length did not match current sample columns "
            f"({len(batch)} vs {len(sample_names)}). {align_note}"
        )
        print("[batch-align] " + note)
        return aligned_batch, note

    raise ValueError(
        f"Length of state.batch ({len(batch)}) does not match number of real sample columns "
        f"({len(sample_names)}), and batch labels could not be realigned. "
        "Check that metadata or batch_info contains 'Batch' plus either 'Sample File', "
        "'Study File ID', 'Sample Name', or 'File Name' columns matching the data columns."
    )


# ---------------------------------------------------------------------
# Spline/CV helpers
# ---------------------------------------------------------------------


def _cubic_spline_smoothing(x, y, s_value, grow=10.0, max_tries=4, tol=1.0):
    """
    Fit a cubic smoothing spline with robust weights from z-scores.

    Returns fitted scipy.interpolate.UnivariateSpline or None.
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import zscore

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if y.size < 4 or np.nanstd(y) == 0:
        return None

    z = zscore(y, nan_policy="omit")
    z = np.where(np.isfinite(z), z, 0.0)
    weights = 1.0 / (1.0 + z**2)

    if np.any(np.diff(x) <= 0):
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        weights = weights[order]

    s_cur = float(s_value)

    for _ in range(max_tries):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")

            try:
                spline = UnivariateSpline(x, y, w=weights, s=s_cur)
            except Exception:
                spline = None

        bad = False

        if wlist:
            text = " ".join(str(w.message) for w in wlist)
            if "maximal number of iterations maxit" in text and "s too small" in text:
                bad = True

        if spline is not None:
            fp = spline.get_residual()

            if s_cur > 0 and np.isfinite(fp):
                ratio = abs(fp - s_cur) / s_cur

                if ratio > tol and s_cur < 1e5:
                    bad = True

        if spline is not None and not bad:
            return spline

        s_cur *= grow

    return None


def _cv_best_smoothing_param(x, y, p_values, k=5, min_points=5, minloo=5):
    """
    Time-aware CV with contiguous folds.

    Falls back to leave-one-out for tiny series.
    """
    from sklearn.model_selection import KFold, LeaveOneOut
    from sklearn.metrics import mean_squared_error

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    n = y.size

    if n < min_points:
        return None

    if np.nanstd(y) == 0:
        return None

    p_values = np.asarray(p_values, dtype=float)
    p_values = p_values[np.isfinite(p_values)]

    if p_values.size == 0:
        return None

    if n < k or k <= 1 or n < minloo:
        splitter = LeaveOneOut()
    else:
        splitter = KFold(n_splits=min(k, n), shuffle=False)

    best_p = None
    best_mse = np.inf

    for p in p_values:
        fold_err = 0.0
        n_splits = 0

        for train_idx, test_idx in splitter.split(x):
            spline = _cubic_spline_smoothing(
                x[train_idx],
                y[train_idx],
                p,
            )

            if spline is None:
                fold_err += 1e12
            else:
                yhat = spline(x[test_idx])
                fold_err += mean_squared_error(y[test_idx], yhat)

            n_splits += 1

        fold_mse = fold_err / max(1, n_splits)

        if fold_mse < best_mse:
            best_mse = fold_mse
            best_p = p

    return best_p


def _estimate_batchwise_s_ranges(data_log2, batch, is_qc_sample, n_features=100, k=5):
    """
    Estimate per-batch smoothing parameter ranges from a subset of features.
    """
    unique_batches = list(dict.fromkeys(batch))
    n_total_features = data_log2.shape[1]

    if n_total_features == 0:
        return {}, {}

    probe_idx = np.linspace(
        0,
        n_total_features - 1,
        min(n_features, n_total_features),
        dtype=int,
    )

    batch_p_ranges = {}
    batch_best_s = {}

    for batch_id in unique_batches:
        is_batch = np.array([b == batch_id for b in batch], dtype=bool)
        qc_mask_batch = np.logical_and(is_qc_sample, is_batch)

        best_s_values = []

        for feature_idx in probe_idx:
            y = data_log2.iloc[qc_mask_batch, feature_idx].values
            x = np.arange(y.size)

            if y.size < 5 or not np.isfinite(y).all() or np.nanstd(y) == 0:
                continue

            var_y = max(np.nanvar(y), 1e-6)
            p_wide = np.logspace(-3, 3, 9) * var_y * y.size
            p_wide = np.clip(p_wide, 1.0, None)

            p_best = _cv_best_smoothing_param(x, y, p_wide, k=k)

            if p_best is not None and np.isfinite(p_best):
                best_s_values.append(p_best)

        batch_best_s[batch_id] = best_s_values

        if len(best_s_values) >= 5:
            s10, _, s90 = np.percentile(best_s_values, [10, 50, 90])
            lower = max(s10 / 3.0, 1e-8)
            upper = s90 * 3.0
            upper = max(upper, lower * 10)
            batch_p_ranges[batch_id] = np.logspace(
                np.log10(lower),
                np.log10(upper),
                7,
            )

        elif len(best_s_values) > 0:
            s50 = float(np.median(best_s_values))
            lower = max(s50 / 10.0, 1e-8)
            upper = max(s50 * 10.0, lower * 10)
            batch_p_ranges[batch_id] = np.logspace(
                np.log10(lower),
                np.log10(upper),
                7,
            )

        else:
            batch_p_ranges[batch_id] = np.logspace(-3, 3, 9)

    return batch_p_ranges, batch_best_s


def _plot_s_exploration(batch_best_s, batch_p_ranges, figures_folder):
    """
    Save diagnostic histograms of selected smoothing parameters.
    """
    import matplotlib.pyplot as plt

    output_paths = []

    for batch_name, s_values in batch_best_s.items():
        if not s_values:
            continue

        s_values = np.asarray(s_values, dtype=float)
        s_values = s_values[np.isfinite(s_values)]

        if s_values.size == 0:
            continue

        p_grid = np.asarray(batch_p_ranges[batch_name], dtype=float)
        p_grid = p_grid[np.isfinite(p_grid)]

        if p_grid.size == 0:
            continue

        log_s = np.log10(s_values)

        s10, s50, s90 = np.percentile(s_values, [10, 50, 90])
        log_lo = np.log10(np.min(p_grid))
        log_hi = np.log10(np.max(p_grid))

        plt.figure(figsize=(8, 4.5))
        bins = max(10, int(np.sqrt(log_s.size)))

        plt.hist(log_s, bins=bins, alpha=0.7, edgecolor="k")

        for value, linestyle, linewidth in [
            (np.log10(s10), "--", 1.0),
            (np.log10(s50), "-", 2.0),
            (np.log10(s90), "--", 1.0),
        ]:
            plt.axvline(value, linestyle=linestyle, linewidth=linewidth)

        plt.axvspan(log_lo, log_hi, alpha=0.15, label="Narrowed range")

        warn = ""

        if (np.log10(s50) - log_lo) < 0.1:
            warn = " (median near LOWER bound)"

        if (log_hi - np.log10(s50)) < 0.1:
            warn = " (median near UPPER bound)"

        plt.xlabel("log10(smoothing parameter s)")
        plt.ylabel("Count")
        plt.title(f"Exploration of best s — batch: {batch_name}{warn}")
        plt.legend(
            [
                f"10/50/90%: {s10:.2g} / {s50:.2g} / {s90:.2g}",
                "Narrowed range",
            ],
            frameon=False,
            loc="best",
        )

        out_path = os.path.join(figures_folder, f"s_exploration_{batch_name}.png")
        out_path = _unique_path(out_path)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[s-explore] Saved: {out_path} (n={s_values.size})")

        output_paths.append(out_path)

    return output_paths


# ---------------------------------------------------------------------
# QC interpolation correction
# ---------------------------------------------------------------------


@register_operation(
    id="correct_qc_interpolation",
    label="QC Interpolation Correction",
    description=(
        "Correct systematic drift using QC-sample spline interpolation. "
        "The correction is fitted in log2 space batch-by-batch and anchored to the global QC median."
    ),
    citation="",
    category_tags=[OperationTag.CORRECTION, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="show",
            type="str_or_list",
            required=False,
            default="default",
            label="Features to plot",
            help="'default', 'all', 'none', an index, cpdID, or list of indexes/cpdIDs.",
        ),
        ParameterDef(
            name="p_values",
            type="str_or_list",
            required=False,
            default="default",
            label="Spline smoothing values",
            help="'default' uses batch-specific CV-estimated ranges, otherwise provide a numeric list.",
        ),
        ParameterDef(
            name="already_log2",
            type="bool",
            required=False,
            default=False,
            label="Already log2 transformed",
            help="If True, input data is treated as already being in log2 space.",
        ),
        ParameterDef(
            name="delog",
            type="bool",
            required=False,
            default=True,
            label="Return raw scale",
            help="If True, back-transform corrected data from log2 to raw intensities.",
        ),
        ParameterDef(
            name="use_zeros",
            type="bool",
            required=False,
            default=False,
            label="Use zeros for fitting",
            help="If False, zero intensities are excluded from QC spline fitting.",
        ),
        ParameterDef(
            name="preserve_zeros",
            type="bool",
            required=False,
            default=True,
            label="Preserve original zeros",
            help="If True, original zero values remain zero after correction.",
        ),
        ParameterDef(
            name="cmap",
            type="str",
            required=False,
            default="viridis",
            label="Batch color map",
            help="Matplotlib colormap name.",
        ),
    ],
    requires=["data", "variable_metadata", "QC_samples"],
    produces=["data", "variable_metadata", "was_log_transformed", "log_base"],
)
def correct_qc_interpolation(
    state: WorkflowState,
    show="default",
    p_values="default",
    already_log2: bool = False,
    delog: bool = True,
    use_zeros: bool = False,
    preserve_zeros: bool = True,
    cmap: str = "viridis",
):
    """
    QC-based drift correction.

    Workflow:
        1. Work in log2 space.
        2. Fit per-batch splines on QC-only points for each feature.
        3. Subtract fitted drift and re-anchor to the global QC median.
        4. Optionally back-transform to raw scale.
    """
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    df_in = state.data

    if df_in is None:
        raise ValueError("No data loaded in state.data.")

    if "cpdID" not in df_in.columns:
        raise ValueError("Expected 'cpdID' column in state.data.")

    variable_metadata = state.variable_metadata

    QC_samples = state.QC_samples

    if QC_samples == []:
        print("There are no QC samples, cannot perform QC-based correction.")

        return {
            "skipped": True,
            "reason": "QC sample list is empty.",
        }

    if QC_samples is None or QC_samples is False:
        raise ValueError(
            "QC samples are not defined or were deleted before, "
            "cannot perform QC-based correction."
        )

    QC_samples = list(QC_samples)

    main_folder, figures_folder = _ensure_main_folders(state)
    suffixes = _get_suffixes(state)

    # Remove generated/non-sample columns before creating the samples x features matrix.
    # This keeps the original correction algorithm intact while preventing stale UI/state
    # columns such as "<1 additional items..." from being treated as samples.
    df_in, sample_columns, dropped_extra_columns = _clean_qc_correction_data_columns(
        state,
        df_in,
    )

    feature_names = df_in["cpdID"].astype(str).reset_index(drop=True)

    # samples x features matrix
    data = df_in[sample_columns].copy().T
    data = data.apply(pd.to_numeric, errors="coerce")

    sample_names = list(data.index)

    # Batch handling.
    batch, batch_alignment_note = _align_batch_to_sample_names(state, sample_names)

    unique_batches = list(dict.fromkeys(batch))

    is_qc_sample = np.array([idx in QC_samples for idx in sample_names], dtype=bool)

    if is_qc_sample.sum() == 0:
        raise ValueError("No QC sample columns found in data.")

    # Batch colors.
    cmap_obj = mpl.colormaps.get_cmap(cmap)
    batch_to_index = {batch_id: i for i, batch_id in enumerate(unique_batches)}
    norm_index = {
        batch_id: i / max(1, len(unique_batches) - 1)
        for batch_id, i in batch_to_index.items()
    }

    batch_to_color = {
        batch_id: mpl.colors.rgb2hex(cmap_obj(norm_index[batch_id]))
        for batch_id in unique_batches
    }

    point_colors = [
        "black" if qc else batch_to_color[batch[i]] for i, qc in enumerate(is_qc_sample)
    ]

    # Original zero mask, before log2 conversion.
    original_zero_mask = data == 0

    if not use_zeros:
        is_zero = original_zero_mask.copy()
    else:
        is_zero = pd.DataFrame(
            False,
            index=data.index,
            columns=data.columns,
        )

    n_features = data.shape[1]
    show_idx = _resolve_show_indices(show, n_features, feature_names)

    if not already_log2:
        eps = 1e-9
        data = np.log2(np.maximum(data, eps))

    # Estimate p grids.
    if isinstance(p_values, str) and p_values == "default":
        batch_p_ranges, batch_best_s = _estimate_batchwise_s_ranges(
            data_log2=data,
            batch=batch,
            is_qc_sample=is_qc_sample,
            n_features=100,
            k=5,
        )

        s_exploration_images = _plot_s_exploration(
            batch_best_s=batch_best_s,
            batch_p_ranges=batch_p_ranges,
            figures_folder=figures_folder,
        )
    else:
        p_values_array = np.asarray(p_values, dtype=float)
        p_values_array = p_values_array[np.isfinite(p_values_array)]

        if p_values_array.size == 0:
            raise ValueError("p_values must be 'default' or a non-empty numeric list.")

        batch_p_ranges = {batch_id: p_values_array for batch_id in unique_batches}
        batch_best_s = {}
        s_exploration_images = []

    start_time = time.time()

    plot_names_original = []
    plot_names_corrected = []
    chosen_p_values = []
    numbers_of_correctable_batches = []

    x_all = np.arange(len(data))

    for feature_idx, feature in enumerate(data.columns):
        splines = []
        is_correctable_batch = []
        num_correctable_batches = 0
        qc_anchor_values = []

        # Fit batch-specific QC splines.
        for batch_idx, batch_name in enumerate(unique_batches):
            is_batch = np.array([b == batch_name for b in batch], dtype=bool)

            qc_mask = is_qc_sample.copy()

            if not use_zeros:
                nonzero_mask = ~is_zero[feature].values
                qc_mask = np.logical_and(qc_mask, nonzero_mask)

            qc_batched = np.logical_and(qc_mask, is_batch)

            qc_y = data.loc[qc_batched, feature]
            x = x_all[qc_batched]
            y = qc_y.values

            p_vals = batch_p_ranges[batch_name]

            if len(y) < 5 or not np.isfinite(y).all() or np.nanstd(y) == 0:
                selected_p = None
            else:
                selected_p = _cv_best_smoothing_param(
                    x,
                    y,
                    p_vals,
                    k=5,
                )

            if selected_p is not None and np.isfinite(selected_p):
                chosen_p_values.append(selected_p)

            if selected_p is None:
                spline = None
                is_correctable_batch.append(False)
            else:
                spline = _cubic_spline_smoothing(x, y, selected_p)
                is_correctable_batch.append(spline is not None)

                if spline is not None:
                    num_correctable_batches += 1
                    qc_anchor_values.append(y[np.isfinite(y)])

            splines.append(spline)

        # Global QC anchor from actually used QC values.
        if len(qc_anchor_values) > 0:
            anchor_log = np.nanmedian(np.concatenate(qc_anchor_values))
        else:
            qc_mask_anchor = is_qc_sample.copy()

            if not use_zeros:
                qc_mask_anchor = np.logical_and(
                    qc_mask_anchor,
                    ~is_zero[feature].values,
                )

            if qc_mask_anchor.any():
                anchor_log = np.nanmedian(data.loc[qc_mask_anchor, feature])
            else:
                anchor_log = 0.0

        numbers_of_correctable_batches.append(num_correctable_batches)

        # BEFORE plot.
        if feature_idx in show_idx:
            zeros_feature = is_zero[feature].values
            alphas = [
                0.4 if qc else (0.1 if z else 0.8)
                for qc, z in zip(is_qc_sample, zeros_feature)
            ]

            plt.figure(figsize=(20, 4))

            y_before = data[feature].values

            plt.scatter(
                x_all,
                y_before,
                color=point_colors,
                alpha=alphas,
            )

            for batch_idx, batch_name in enumerate(unique_batches):
                is_batch = np.array([b == batch_name for b in batch], dtype=bool)
                xb = x_all[is_batch]
                spline = splines[batch_idx]

                if spline is not None:
                    plt.plot(
                        xb,
                        spline(xb),
                        linewidth=2,
                        label=f"spline {batch_name}",
                    )

            plt.axhline(
                anchor_log,
                color="0.5",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                label="QC anchor",
            )

            plt.xlabel("Injection Order")
            plt.ylabel("log2 Peak Area before correction")
            plt.title(str(feature_names.iloc[feature_idx]))
            plt.legend(loc="best", frameon=False)

            plt_name = os.path.join(
                figures_folder,
                f"QC_correction_{feature_idx}_original",
            )

            saved_png = None
            first_saved = None

            for suffix in suffixes:
                # plt.savefig(plt_name + suffix, dpi=300, bbox_inches="tight")
                out_path = _unique_path(plt_name + suffix)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")

                if first_saved is None:
                    first_saved = out_path

                if suffix == ".png":
                    saved_png = out_path

            # plot_names_original.append(plt_name + ".png")
            plot_names_original.append(
                saved_png if saved_png is not None else first_saved
            )
            plt.close()

        # Apply correction in log space.
        for batch_idx, batch_name in enumerate(unique_batches):
            is_batch = np.array([b == batch_name for b in batch], dtype=bool)
            spline = splines[batch_idx]

            if spline is None:
                continue

            preds = spline(x_all[is_batch])

            data.loc[is_batch, feature] = (
                data.loc[is_batch, feature] - preds + anchor_log
            )

        # AFTER plot.
        if feature_idx in show_idx:
            zeros_feature = is_zero[feature].values
            alphas = [
                0.4 if qc else (0.1 if z else 0.8)
                for qc, z in zip(is_qc_sample, zeros_feature)
            ]

            plt.figure(figsize=(20, 4))

            y_corrected = data[feature].values

            plt.scatter(
                x_all,
                y_corrected,
                color=point_colors,
                alpha=alphas,
            )

            for batch_idx, batch_name in enumerate(unique_batches):
                is_batch = np.array([b == batch_name for b in batch], dtype=bool)
                xb = x_all[is_batch]
                spline = splines[batch_idx]

                if spline is not None:
                    yb_original = spline(xb)
                    yb_corrected = yb_original - yb_original + anchor_log

                    # preds_min = min(preds_min, np.nanmin(yb_corrected))
                    # preds_max = max(preds_max, np.nanmax(yb_corrected))

                    plt.plot(
                        xb,
                        yb_corrected,
                        color="green",
                        linewidth=2,
                        alpha=0.8,
                        label="corrected spline" if batch_idx == 0 else None,
                    )

            plt.axhline(
                anchor_log,
                color="0.5",
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                label="QC anchor",
            )

            plt.xlabel("Injection Order")
            plt.ylabel("log2 Peak Area after correction")
            plt.title(str(feature_names.iloc[feature_idx]))
            plt.legend(loc="best", frameon=False)

            plt_name = os.path.join(
                figures_folder,
                f"QC_correction_{feature_idx}_corrected",
            )

            for suffix in suffixes:
                # plt.savefig(plt_name + suffix, dpi=300, bbox_inches="tight")
                out_path = _unique_path(plt_name + suffix)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")

                if first_saved is None:
                    first_saved = out_path

                if suffix == ".png":
                    saved_png = out_path

            plot_names_corrected.append(plt_name + ".png")
            plt.close()

        pct = round((feature_idx + 1) / n_features * 100, 3)
        eta_min = round(
            (time.time() - start_time)
            / (feature_idx + 1)
            * (n_features - feature_idx - 1)
            / 60,
            2,
        )

        print(
            f"Progress: {pct}%; last feature: {feature}. ETA ~ {eta_min}m       ",
            end="\r",
        )

    print("")

    # Back-transform if requested.
    if delog:
        data = np.power(2.0, data)

        if preserve_zeros:
            data[original_zero_mask] = 0

        state.was_log_transformed = False
        state.log_base = None
    else:
        # corrected data stays in log2 space; do not force raw zeros into log2 zero
        print("Data was kept in log2 space;-not forcing zeros.")
        state.was_log_transformed = True
        state.log_base = 2

    # Return to features x samples with cpdID.
    data = data.T
    data.insert(0, "cpdID", feature_names.values)

    valid_p_values = [p for p in chosen_p_values if p is not None and np.isfinite(p)]

    if valid_p_values:
        chosen_p_values_summary = pd.Series(valid_p_values).value_counts().to_dict()
        chosen_p_values_summary = {
            k: v
            for k, v in sorted(
                chosen_p_values_summary.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
    else:
        chosen_p_values_summary = {}

    correction_dict = {
        cpdID: n_corrected
        for cpdID, n_corrected in zip(
            feature_names,
            numbers_of_correctable_batches,
        )
    }

    if variable_metadata is not None and "cpdID" in variable_metadata.columns:
        variable_metadata = variable_metadata.copy()
        variable_metadata["corrected_batches"] = variable_metadata["cpdID"].map(
            correction_dict
        )

    state.data = data
    state.variable_metadata = _filter_match_variable_metadata(
        data,
        variable_metadata,
    )

    # REPORTING
    add_text(
        state,
        f"Correction was performed in log2 space " + "but the data were transformed back after the correction." if delog else "and were kept that way on the output.",
        title = "Details")

    for i in range(len(plot_names_original)): #should be same lengths
        path_original = plot_names_original[i]
        path_corrected = plot_names_corrected[i]
        add_figure(
            state,
            path_original,
            title="Before correction",
            caption="Example feature #"+ str(i+1) +" before correction.",
        )
        add_figure(
                    state,
                    path_corrected,
                    title="After correction",
                    caption="Example feature #"+ str(i+1) +" after correction.",
                )

    return {
        "features_corrected": int(n_features),
        "batches": list(unique_batches),
        "chosen_p_values": chosen_p_values_summary,
        "s_exploration_images": s_exploration_images,
        "before_plots": plot_names_original,
        "after_plots": plot_names_corrected,
        "output_space": "raw" if delog else "log2",
        "was_log_transformed": state.was_log_transformed,
        "log_base": state.log_base,
        "dropped_non_sample_columns": dropped_extra_columns,
        "batch_alignment_note": batch_alignment_note,
    }


# ---------------------------------------------------------------------
# ComBat correction
# ---------------------------------------------------------------------


def _load_pycombat():
    """
    Load pyComBat implementation lazily.

    Different installations expose it differently, so this tries common imports.
    """
    try:
        from combat.pycombat import pycombat

        return pycombat
    except Exception:
        pass

    try:
        from pycombat import pycombat

        return pycombat
    except Exception:
        pass

    raise ImportError(
        "Could not import pycombat. Install the package that provides "
        "`combat.pycombat.pycombat` or adjust _load_pycombat() to your installed package."
    )


def _align_metadata_to_samples(metadata, sample_cols):
    """
    Align metadata rows to sample columns.

    If metadata has 'Sample File', use it.
    Otherwise assume metadata is already in sample-column order.
    """
    if metadata is None:
        return None

    if "Sample File" in metadata.columns:
        missing = [
            sample
            for sample in sample_cols
            if sample not in set(metadata["Sample File"])
        ]

        if missing:
            raise ValueError(
                "Some data sample columns are missing from metadata['Sample File']: "
                + str(missing[:10])
            )

        return metadata.set_index("Sample File").loc[sample_cols].reset_index()

    if len(metadata) != len(sample_cols):
        raise ValueError(
            "metadata has no 'Sample File' column and its length does not match "
            "the number of sample columns."
        )

    return metadata.reset_index(drop=True).copy()


@register_operation(
    id="correct_combat",
    label="ComBat Batch Correction",
    description=(
        "Correct batch effects using ComBat. Intended mainly for datasets without QC samples, "
        "where known batch labels are available."
    ),
    citation="",
    category_tags=[OperationTag.CORRECTION],
    parameter_schema=[
        ParameterDef(
            name="par_prior",
            type="bool",
            required=False,
            default=True,
            label="Parametric prior",
            help="Passed to pycombat.",
        ),
        ParameterDef(
            name="mean_only",
            type="bool",
            required=False,
            default=False,
            label="Mean-only adjustment",
            help="If True, only batch mean differences are corrected.",
        ),
        ParameterDef(
            name="ref_batch",
            type="str_or_none",
            required=False,
            default=None,
            label="Reference batch",
            help="Optional reference batch ID.",
        ),
        ParameterDef(
            name="already_log2",
            type="bool",
            required=False,
            default=False,
            label="Already log2 transformed",
            help="If True, input data is treated as already log2 transformed.",
        ),
        ParameterDef(
            name="delog",
            type="bool",
            required=False,
            default=False,
            label="Return raw scale",
            help="If True, back-transform corrected data from log2 to raw intensities.",
        ),
        ParameterDef(
            name="covariate_metadata_columns",
            type="list_or_none",
            required=False,
            default=None,
            label="Covariate metadata columns",
            help="Metadata columns to preserve during ComBat adjustment.",
        ),
    ],
    requires=["data", "batch"],
    produces=["data", "was_log_transformed", "log_base"],
)
def correct_combat(
    state: WorkflowState,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch=None,
    already_log2: bool = False,
    delog: bool = False,
    covariate_metadata_columns=None,
):
    """
    Correct batch effects using ComBat.

    Important:
        ComBat needs batch labels aligned to sample columns.
        If covariates are used, metadata must also align to sample columns.
    """
    pycombat = _load_pycombat()

    data = state.data

    if data is None:
        raise ValueError("No data loaded in state.data.")

    if "cpdID" not in data.columns:
        raise ValueError("Expected a 'cpdID' column in state.data.")

    batch = getattr(state, "batch", None)

    if batch is None:
        raise ValueError(
            "Batch information is None. Set state.batch before running ComBat."
        )

    batch = list(batch)

    sample_cols = list(data.columns[1:])
    n_samples = len(sample_cols)

    if len(batch) != n_samples:
        raise ValueError(
            f"Length of state.batch ({len(batch)}) does not match "
            f"the number of sample columns ({n_samples})."
        )

    unique_batches = pd.Series(batch).dropna().unique()

    if len(unique_batches) < 2:
        raise ValueError(
            "ComBat requires at least 2 different batches, "
            f"but found only: {list(unique_batches)}."
        )

    expr = data.iloc[:, 1:].copy()
    expr = expr.apply(pd.to_numeric, errors="coerce")

    if expr.isna().any().any():
        bad_cols = expr.columns[expr.isna().any()].tolist()
        raise ValueError(
            "ComBat input contains missing/non-numeric values after numeric coercion. "
            "Impute or filter missing values before ComBat. Example problematic columns: "
            + str(bad_cols[:10])
        )

    if not already_log2:
        eps = 1e-9
        expr = np.log2(np.maximum(expr, eps))
        expr = pd.DataFrame(
            expr,
            index=data.index,
            columns=sample_cols,
        )

    # Covariates.
    mod = []

    if covariate_metadata_columns is not None:
        if isinstance(covariate_metadata_columns, str):
            covariate_metadata_columns = [covariate_metadata_columns]

        if len(covariate_metadata_columns) > 0:
            metadata = _align_metadata_to_samples(
                state.metadata,
                sample_cols,
            )

            if metadata is None:
                raise ValueError(
                    "covariate_metadata_columns were provided, but state.metadata is None."
                )

            for col in covariate_metadata_columns:
                if col not in metadata.columns:
                    raise ValueError(
                        f"Covariate metadata column '{col}' not found in metadata."
                    )

                covariate_values = metadata[col].tolist()

                if len(covariate_values) != n_samples:
                    raise ValueError(
                        f"Length of covariate column '{col}' does not match "
                        f"the number of sample columns ({n_samples})."
                    )

                mod.append(covariate_values)

    expr_corrected = pycombat(
        data=expr,
        batch=batch,
        mod=mod,
        par_prior=par_prior,
        mean_only=mean_only,
        ref_batch=ref_batch,
    )

    if isinstance(expr_corrected, pd.DataFrame):
        expr_corrected_df = expr_corrected.copy()
        expr_corrected_df.index = expr.index
        expr_corrected_df.columns = expr.columns
    else:
        expr_corrected_df = pd.DataFrame(
            expr_corrected,
            index=expr.index,
            columns=expr.columns,
        )

    df_out = data.copy()
    df_out.iloc[:, 1:] = expr_corrected_df

    if delog:
        df_out.iloc[:, 1:] = np.power(2.0, df_out.iloc[:, 1:])
        state.was_log_transformed = False
        state.log_base = None
        output_space = "raw"
    else:
        state.was_log_transformed = True
        state.log_base = 2
        output_space = "log2"

    state.data = df_out

    print("ComBat batch correction applied.")
    print(f"Data kept in {output_space} space.")


    # REPORTING
    add_text(
        state,
        f"Correction was performed in log2 space " + "but the data were transformed back after the correction." if delog else "and were kept that way on the output.",
        title = "Details")

    return {
        "corrected": True,
        "method": "ComBat",
        "n_samples": n_samples,
        "n_features": int(data.shape[0]),
        "batches": list(unique_batches),
        "par_prior": par_prior,
        "mean_only": mean_only,
        "ref_batch": ref_batch,
        "covariates": covariate_metadata_columns,
        "output_space": output_space,
        "was_log_transformed": state.was_log_transformed,
        "log_base": state.log_base,
    }
