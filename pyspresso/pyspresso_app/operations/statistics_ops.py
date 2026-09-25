# pyspresso_app/operations/statistics_ops.py

import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from itertools import cycle

from scipy.stats import shapiro, ttest_ind, mannwhitneyu, zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_auc_score

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState
from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _unique_path(path: str):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1

    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _natural_sort_key(s):
    import re

    if isinstance(s, tuple):
        s = "-".join(map(str, s))

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def _load_plotting():
    """
    Import plotting libraries lazily so operation registration does not fail
    if plotting dependencies are not installed yet.
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for statistics operations. "
            "Install them with: python -m pip install matplotlib seaborn"
        ) from exc

    return plt, sns


def _ensure_statistics_folder(state: WorkflowState):
    main_folder = getattr(state, "main_folder", None)

    if main_folder is None:
        workflow_id = getattr(state, "workflow_id", "workflow")
        main_folder = os.path.join("outputs", str(workflow_id))
        state.main_folder = main_folder

    statistics_folder = os.path.join(main_folder, "statistics")
    os.makedirs(main_folder, exist_ok=True)
    os.makedirs(statistics_folder, exist_ok=True)

    return main_folder, statistics_folder


def _get_output_file_prefix(state: WorkflowState):
    output_file_prefix = getattr(state, "output_file_prefix", None)

    if output_file_prefix is None:
        output_file_prefix = getattr(state, "name", "PySPRESSO_Workflow")
        state.output_file_prefix = output_file_prefix

    return output_file_prefix


def _get_suffixes(state: WorkflowState):
    suffixes = getattr(state, "suffixes", None)

    if suffixes is None:
        suffixes = [".png"]
        state.suffixes = suffixes

    return suffixes


def _add_artifact_if_available(state: WorkflowState, path, artifact_type, description):
    artifacts = getattr(state, "artifacts", None)

    if artifacts is None:
        return

    artifacts.append(
        {
            "type": artifact_type,
            "path": path,
            "description": description,
        }
    )


def _candidates_calculate_hits(state: WorkflowState):
    hits = []

    for feature in state.candidates["feature"]:
        hit = state.candidates[state.candidates["feature"] == feature].shape[0]
        hits.append(hit)

    state.candidates["hits"] = hits
    return state.candidates


def _candidates_order(state: WorkflowState, how="method"):
    if how == "method":
        state.candidates = state.candidates.sort_values(
            by=["method", "specification", "score", "hits"],
            ascending=[False, False, False, False],
        )
    elif how == "hits":
        state.candidates = state.candidates.sort_values(
            by=["hits", "method", "specification", "score"],
            ascending=[False, False, False, False],
        )

    state.candidates.reset_index(drop=True, inplace=True)
    return state.candidates


def _add_candidates(state: WorkflowState, features, method, specification, scores):
    """
    Port of old Workflow.add_candidates(), adapted to WorkflowState.
    """
    variable_metadata = state.variable_metadata

    if not isinstance(features, (list, pd.Series, np.ndarray)):
        features = [features]
    if not isinstance(scores, (list, pd.Series, np.ndarray)):
        scores = [scores]

    data = state.data
    if data is None:
        raise ValueError("state.data is None; cannot map candidate indices to cpdIDs.")

    if "cpdID" in data.columns:
        id_series = data["cpdID"].astype(str).reset_index(drop=True)
    else:
        id_series = data.iloc[:, 0].astype(str).reset_index(drop=True)

    normalized_features = []
    for feature in features:
        if isinstance(feature, (int, np.integer)):
            idx = int(feature)
            if idx < 0 or idx >= len(id_series):
                raise ValueError(
                    f"Candidate index {idx} out of range (0..{len(id_series) - 1})."
                )
            normalized_features.append(id_series.iloc[idx])
        else:
            normalized_features.append(str(feature))

    features = normalized_features

    if isinstance(method, list) and len(method) != len(features):
        raise ValueError("Length of method list must match length of features.")
    if isinstance(specification, list) and len(specification) != len(features):
        raise ValueError("Length of specification list must match length of features.")

    if isinstance(method, str) or (
        isinstance(method, (list, pd.Series, np.ndarray)) and len(method) == 1
    ):
        method = [method] * len(features)
    if isinstance(specification, str) or (
        isinstance(specification, (list, pd.Series, np.ndarray))
        and len(specification) == 1
    ):
        specification = [specification] * len(features)

    hits = [0] * len(features)

    if state.candidates is None:
        state.candidates = pd.DataFrame(
            columns=[
                "feature",
                "method",
                "specification",
                "score",
                "hits",
                "Name",
                "Formula",
                "Annot. DeltaMass [ppm]",
                "Annotation MW",
            ]
        )

    candidate_columns = {
        "feature": features,
        "method": method,
        "specification": specification,
        "score": scores,
        "hits": hits,
    }

    if variable_metadata is not None and "cpdID" in variable_metadata.columns:
        metadata_lookup = variable_metadata.copy()
        metadata_lookup["cpdID"] = metadata_lookup["cpdID"].astype(str)
        metadata_lookup = metadata_lookup.drop_duplicates("cpdID").set_index("cpdID")

        name_column = (
            "Name" if "Name" in metadata_lookup.columns else "Compound Name"
        )
        optional_columns = [
            ("Name", name_column),
            ("Formula", "Formula"),
            ("Annot. DeltaMass [ppm]", "Annot. DeltaMass [ppm]"),
            ("Annotation MW", "Annotation MW"),
        ]
        for output_column, source_column in optional_columns:
            if source_column in metadata_lookup.columns:
                candidate_columns[output_column] = metadata_lookup.reindex(features)[
                    source_column
                ].tolist()

    new_candidates = pd.DataFrame(candidate_columns)

    if state.candidates.empty:
        state.candidates = new_candidates
    else:
        state.candidates = pd.concat(
            [state.candidates, new_candidates], ignore_index=True
        )

    state.candidates = _candidates_calculate_hits(state)
    state.candidates = _candidates_order(state, how="method")

    return state.candidates


def _shapiro_ok(x):
    """
    Port of old Workflow._shapiro_ok().
    """
    if x.size < 3 or np.all(x == x[0]):
        return "Not enough data for normality test"
    stat, p = shapiro(x)
    return p > 0.05


def _adjust_pvalues(p_values, method=None):
    """Apply a supported multiple-testing correction while preserving NaNs."""
    normalized_method = "none" if method is None else str(method).strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "fdr_bh": "fdr_bh",
        "bh": "fdr_bh",
        "benjamini-hochberg": "fdr_bh",
        "benjamini_hochberg": "fdr_bh",
        "bonferroni": "bonferroni",
    }
    if normalized_method not in aliases:
        raise ValueError(
            "p_value_correction_method must be None, 'fdr_bh', or 'bonferroni'."
        )
    normalized_method = aliases[normalized_method]

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not len(valid_indices):
        return adjusted, normalized_method

    valid = values[valid_indices]
    if normalized_method == "none":
        adjusted[valid_indices] = valid
    elif normalized_method == "bonferroni":
        adjusted[valid_indices] = np.minimum(valid * len(valid), 1.0)
    else:
        order = np.argsort(valid)
        ranked = valid[order]
        corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        restored = np.empty_like(corrected)
        restored[order] = np.minimum(corrected, 1.0)
        adjusted[valid_indices] = restored

    return adjusted, normalized_method


def _plsda_double_cv_predict(
    X,
    Y,
    y_strat,
    outer_splits=5,
    outer_repeats=10,
    inner_splits=5,
    ncomp_grid=None,
    select_metric="auroc",
    rng=42,
):
    """
    Port of old Workflow._plsda_double_cv_predict().
    """
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]

    warned_caps = set()

    if ncomp_grid is None:
        max_lv = min(10, n_features, n_samples - 1)
        ncomp_grid = list(range(1, max_lv + 1))
    else:
        max_lv = min(n_features, n_samples - 1)
        ncomp_grid = [int(k) for k in ncomp_grid if 1 <= int(k) <= max_lv]
        if len(ncomp_grid) == 0:
            raise ValueError("ncomp_grid is empty after applying validity constraints.")

    pred_sum = np.zeros((n_samples, n_classes), dtype=float)
    pred_count = np.zeros((n_samples,), dtype=int)
    repeat_predictions = []

    chosen_lvs = []

    def _score(Y_true, Y_pred):
        if select_metric == "auroc":
            if Y_true.shape[1] == 2:
                return float(roc_auc_score(Y_true[:, 1], Y_pred[:, 1]))
            return float(
                roc_auc_score(Y_true, Y_pred, multi_class="ovr", average="macro")
            )
        elif select_metric == "nmc":
            yt = Y_true.argmax(axis=1)
            yp = Y_pred.argmax(axis=1)
            return float((yt != yp).sum())
        else:
            raise ValueError("select_metric must be 'auroc' or 'nmc'")

    for rep in range(int(outer_repeats)):
        repeat_pred = np.full((n_samples, n_classes), np.nan, dtype=float)
        _, cts = np.unique(y_strat, return_counts=True)
        outer_min_class = int(cts.min()) if cts.size else 0
        outer_k = min(int(outer_splits), outer_min_class)
        if outer_k < 2:
            raise ValueError(
                "Not enough samples per class to run outer StratifiedKFold "
                "(need >=2). You can ignore these groups by using the "
                "'ignored_groups' parameter in statistics_PLSDA()."
            )
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=rng + rep)

        for tr_outer, te_outer in outer.split(X, y_strat):
            X_tr, X_te = X[tr_outer], X[te_outer]

            max_lv_outer = min(10, X_tr.shape[1], X_tr.shape[0] - 1)
            lv_grid_outer = [lv for lv in ncomp_grid if lv <= max_lv_outer]
            if not lv_grid_outer:
                lv_grid_outer = [1]

            Y_tr = Y[tr_outer]
            y_tr_strat = y_strat[tr_outer]

            unique, counts = np.unique(y_tr_strat, return_counts=True)
            inner_min_class = int(counts.min()) if counts.size else 0

            if inner_min_class < 2:
                best_lv = lv_grid_outer[0]
            else:
                inner_k = min(int(inner_splits), inner_min_class)
                inner = StratifiedKFold(
                    n_splits=inner_k, shuffle=True, random_state=rng + 1000 + rep
                )

                min_inner_train = min(
                    len(tr_in) for tr_in, _ in inner.split(X_tr, y_tr_strat)
                )
                max_lv_inner = min(10, X_tr.shape[1], min_inner_train - 1)
                lv_grid = [lv for lv in lv_grid_outer if lv <= max_lv_inner] or [
                    lv_grid_outer[0]
                ]

                cap_key = (X_tr.shape[0], min_inner_train, max(lv_grid))
                if max(lv_grid) < max(lv_grid_outer) and cap_key not in warned_caps:
                    warnings.warn(
                        f"PLS-DA: LV grid capped to {max(lv_grid)} due to fold size limits "
                        f"(outer train n={X_tr.shape[0]}, inner train min n={min_inner_train}).",
                        UserWarning,
                    )
                    warned_caps.add(cap_key)

                best_lv = None
                best_score = None

                for lv in lv_grid:
                    y_inner = np.zeros_like(Y_tr, dtype=float)

                    feasible = True
                    for tr_in, te_in in inner.split(X_tr, y_tr_strat):
                        max_lv_this = min(10, X_tr.shape[1], len(tr_in) - 1)
                        if lv > max_lv_this:
                            feasible = False
                            break

                        # Upstream PySPRESSO scaling (including Pareto scaling)
                        # must not be replaced by sklearn's unit-variance scaling.
                        m = PLSRegression(n_components=int(lv), scale=False)
                        m.fit(X_tr[tr_in], Y_tr[tr_in])
                        y_inner[te_in] = m.predict(X_tr[te_in])

                    if not feasible:
                        continue

                    try:
                        s = _score(Y_tr, y_inner)
                    except ValueError as exc:
                        warnings.warn(
                            f"Inner-CV scoring failed for lv={lv}: {exc}", UserWarning
                        )
                        continue

                    if best_lv is None:
                        best_lv, best_score = lv, s
                    else:
                        if select_metric == "auroc":
                            if s > best_score:
                                best_lv, best_score = lv, s
                        else:
                            if s < best_score:
                                best_lv, best_score = lv, s

                if best_lv is None:
                    best_lv = lv_grid[0]

            chosen_lvs.append(int(best_lv))

            m_outer = PLSRegression(n_components=int(best_lv), scale=False)
            m_outer.fit(X_tr, Y_tr)
            y_hat = m_outer.predict(X_te)

            pred_sum[te_outer] += y_hat
            pred_count[te_outer] += 1
            repeat_pred[te_outer] = y_hat

        if np.isnan(repeat_pred).any():
            raise RuntimeError(
                "PLS-DA outer cross-validation did not predict every sample."
            )
        repeat_predictions.append(repeat_pred)

    y_cv_outer = pred_sum / np.maximum(pred_count[:, None], 1)

    return y_cv_outer, chosen_lvs, np.stack(repeat_predictions, axis=0)


def _vip(model):
    """
    Port of old Workflow._vip().
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    if not np.isfinite(total_s) or total_s <= np.finfo(float).eps:
        return vips
    for i in range(p):
        weight = np.array(
            [
                (w[i, j] / norm) ** 2 if norm > np.finfo(float).eps else 0.0
                for j in range(h)
                for norm in [np.linalg.norm(w[:, j])]
            ]
        )
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips


def _classification_metrics(Y_true, Y_pred):
    """Return classification metrics for one complete CV prediction matrix."""
    true_codes = Y_true.argmax(axis=1)
    predicted_codes = Y_pred.argmax(axis=1)
    nmc = int((true_codes != predicted_codes).sum())
    accuracy = float((true_codes == predicted_codes).mean())

    try:
        if Y_true.shape[1] == 2:
            auc = float(roc_auc_score(Y_true[:, 1], Y_pred[:, 1]))
        else:
            auc = float(
                roc_auc_score(Y_true, Y_pred, multi_class="ovr", average="macro")
            )
    except ValueError:
        auc = np.nan

    return nmc, accuracy, auc


def _q2(Y_true, predictions):
    """Calculate repeated-CV Q2 per class and globally from held-out predictions."""
    repeated_truth = np.broadcast_to(Y_true, predictions.shape)
    press_per_class = np.sum((repeated_truth - predictions) ** 2, axis=(0, 1))
    centered = Y_true - Y_true.mean(axis=0, keepdims=True)
    tss_per_class = predictions.shape[0] * np.sum(centered**2, axis=0)
    q2_per_class = np.divide(
        press_per_class,
        tss_per_class,
        out=np.full_like(press_per_class, np.nan, dtype=float),
        where=tss_per_class > 0,
    )
    q2_per_class = 1.0 - q2_per_class

    flat_truth = repeated_truth.ravel()
    flat_predictions = predictions.ravel()
    global_tss = np.sum((flat_truth - flat_truth.mean()) ** 2)
    q2_global = (
        float(1.0 - np.sum((flat_truth - flat_predictions) ** 2) / global_tss)
        if global_tss > 0
        else np.nan
    )
    return q2_per_class, q2_global


# ---------------------------------------------------------------------
# Statistics operations
# ---------------------------------------------------------------------


@register_operation(
    id="statistics_correlation_means",
    label="Correlation of Group Means",
    description="Calculate the correlation matrix of group means and create a heatmap.",
    citation="",
    category_tags=[OperationTag.STATISTICS],
    parameter_schema=[
        ParameterDef(
            name="column_name",
            type="str",
            required=True,
            default=None,
            label="Grouping metadata column",
            help="Metadata column used to group samples, e.g. Sample Type, Sex, Diagnosis.",
        ),
        ParameterDef(
            name="method",
            type="str",
            required=False,
            default="pearson",
            label="Correlation method",
            help="Accepted for compatibility with old PySPRESSO. Old function did not pass this argument into corr().",
        ),
        ParameterDef(
            name="cmap",
            type="str",
            required=False,
            default="coolwarm",
            label="Colormap",
        ),
        ParameterDef(
            name="min_max",
            type="list",
            required=False,
            default=[-1, 1],
            label="Color scale min/max",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
        ),
    ],
    requires=["data", "metadata"],
    produces=["statistics"],
)
def statistics_correlation_means(
    state: WorkflowState,
    column_name,
    method="pearson",
    cmap="coolwarm",
    min_max=[-1, 1],
    plt_name_suffix="",
):
    plt, sns = _load_plotting()

    data = state.data
    metadata = state.metadata
    output_file_prefix = _get_output_file_prefix(state)
    main_folder, statistics_folder = _ensure_statistics_folder(state)
    suffixes = _get_suffixes(state)

    if data is None:
        raise ValueError("No data loaded in state.data.")
    if metadata is None:
        raise ValueError("No metadata loaded in state.metadata.")
    if column_name not in metadata.columns:
        raise ValueError(f"Column '{column_name}' was not found in metadata.")
    if "Sample File" not in metadata.columns:
        raise ValueError("Expected 'Sample File' in metadata to align samples.")

    method = str(method).strip().lower()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")

    if not isinstance(min_max, (list, tuple)) or len(min_max) != 2:
        raise ValueError("min_max must contain exactly [minimum, maximum].")
    min_max = [float(min_max[0]), float(min_max[1])]

    if min_max[0] < -1:
        min_max[0] = -1
    if min_max[1] > 1:
        min_max[1] = 1
    if min_max[0] > min_max[1]:
        min_max = [-1, 1]

    sample_names = metadata["Sample File"].astype(str).tolist()
    missing_samples = [sample for sample in sample_names if sample not in data.columns]
    if missing_samples:
        raise ValueError(
            "Metadata sample(s) missing from data: "
            + ", ".join(missing_samples[:10])
        )

    try:
        sample_by_feature = data.set_index("cpdID")[sample_names].T.apply(
            pd.to_numeric, errors="raise"
        )
    except KeyError as exc:
        raise ValueError("Expected a 'cpdID' column in data.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("Correlation input contains non-numeric abundance values.") from exc

    sample_by_feature.index = sample_names
    groups = metadata.set_index(metadata["Sample File"].astype(str))[column_name]
    grouped_means = sample_by_feature.groupby(groups, dropna=True).mean()
    if len(grouped_means) < 2:
        raise ValueError("At least two non-empty groups are required for correlation.")

    correlation_matrix = grouped_means.T.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap=cmap, vmin=min_max[0], vmax=min_max[1], ax=ax)

    name = os.path.join(
        statistics_folder,
        output_file_prefix
        + "_group_correlation_matrix_heatmap_"
        + str(plt_name_suffix),
    )

    saved_paths = []
    image_for_report = None

    for suffix in suffixes:
        out_path = _unique_path(name + suffix)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        saved_paths.append(out_path)
        if suffix == ".png":
            image_for_report = out_path

    if image_for_report is None and saved_paths:
        image_for_report = saved_paths[0]

    plt.close(fig)

    csv_path = _unique_path(name + ".csv")
    correlation_matrix.to_csv(csv_path, sep=";")
    _add_artifact_if_available(
        state,
        csv_path,
        "table",
        "Group mean correlation matrix",
    )

    if image_for_report is not None:
        _add_artifact_if_available(
            state,
            image_for_report,
            "figure",
            "Group mean correlation matrix heatmap",
        )
        
    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Correlation of group mean feature profiles was calculated using "
            f"the '{method}' correlation method. "
            f"Samples were grouped according to metadata column '{column_name}'."
        ),
        title="Correlation of group means",
    )

    add_table(
        state,
        correlation_matrix,
        title="Group correlation matrix",
        include_index=True,
    )

    if image_for_report is not None:
        add_figure(
            state,
            image_for_report,
            title="Group correlation matrix heatmap",
        )

    return {
        "message": "Group correlation matrix heatmap was created.",
        "column_name": column_name,
        "method": method,
        "saved_paths": saved_paths,
        "table_path": csv_path,
    }


@register_operation(
    id="statistics_PCA",
    label="PCA",
    description="Perform PCA on the data and store PCA scores, loadings, explained variance, and candidate features.",
    citation="",
    category_tags=[OperationTag.STATISTICS],
    parameter_schema=[
        ParameterDef(
            name="n_components_for_candidates",
            type="int",
            required=False,
            default=2,
            label="Components for candidates",
            help="Number of first PCA components used for candidate-feature selection.",
        ),
    ],
    requires=["data"],
    produces=["pca", "pca_data", "pca_df", "pca_per_var", "pca_loadings", "candidates"],
)
def statistics_PCA(state: WorkflowState, n_components_for_candidates=2):
    data = state.data

    if data is None:
        raise ValueError("No data loaded in state.data.")

    was_centered = state.was_centered
    was_scaled = state.was_scaled

    warning_messages = []

    if not was_centered:
        msg = "Data was not centered. It's highly recommended to center the data before performing PCA."
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    if not was_scaled:
        msg = "Data was not scaled. It's suggested to scale the data before performing PCA."
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    pca = PCA()
    pca.fit(data.iloc[:, 1:].T)
    pca_data = pca.transform(data.iloc[:, 1:].T)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)

    state.pca_per_var = per_var.tolist()
    state.pca_data = pca_data
    state.pca_df = pca_df
    state.pca = pca

    state.pca_count += 1

    eigenvectors = pca.components_.T
    eigenvalues = pca.explained_variance_
    loadings = eigenvectors * np.sqrt(eigenvalues)

    loadings_df = pd.DataFrame(loadings, index=data.iloc[:, 0], columns=labels)
    state.pca_loadings = loadings_df

    if n_components_for_candidates > len(labels):
        n_components_for_candidates = len(labels)

    loadings_df["distance"] = np.sqrt(
        np.sum(loadings_df.iloc[:, :n_components_for_candidates] ** 2, axis=1)
    )
    loadings_df["distance"] = loadings_df["distance"] / np.max(loadings_df["distance"])
    loadings_df["distance"] = loadings_df["distance"] * 100

    candidate_loadings = loadings_df[
        loadings_df["distance"] > np.percentile(loadings_df["distance"], 99.5)
    ].index
    candidate_loadings_scores = loadings_df[
        loadings_df["distance"] > np.percentile(loadings_df["distance"], 99.5)
    ]["distance"].round(2)

    state.pca_loadings_candidates = candidate_loadings.to_list()
    state.pca_loadings_candidates_len = len(candidate_loadings)

    _add_candidates(
        state=state,
        features=candidate_loadings.to_list(),
        method="PCA-loadings",
        specification="Analysis-"
        + str(state.pca_count)
        + "; PCA-components-"
        + str(n_components_for_candidates),
        scores=candidate_loadings_scores.to_list(),
    )

    main_folder, statistics_folder = _ensure_statistics_folder(state)
    output_file_prefix = _get_output_file_prefix(state)

    scores_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PCA_scores.csv")
    )
    loadings_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PCA_loadings.csv")
    )
    candidates_path = _unique_path(
        os.path.join(
            statistics_folder, output_file_prefix + "_candidates_after_PCA.csv"
        )
    )

    pca_df.to_csv(scores_path, index=False, sep=";")
    loadings_df.to_csv(loadings_path, sep=";")
    state.candidates.to_csv(candidates_path, index=False, sep=";")

    _add_artifact_if_available(state, scores_path, "table", "PCA scores")
    _add_artifact_if_available(state, loadings_path, "table", "PCA loadings")
    _add_artifact_if_available(
        state, candidates_path, "table", "Candidate features after PCA"
    )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"PCA analysis {state.pca_count} was performed on "
            f"{data.shape[0]} features and {data.shape[1] - 1} samples. "
            f"{len(candidate_loadings)} candidate features were identified "
            f"from the first {n_components_for_candidates} principal components."
        ),
        title=f"PCA analysis {state.pca_count}",
    )

    explained_variance_table = pd.DataFrame(
    {
        "Component": labels,
        "Explained variance (%)": per_var,
    }
    )

    add_table(
        state,
        explained_variance_table,
        title="Explained variance",
        include_index=False,
        max_rows=20,
    )
    if len(candidate_loadings) > 0:
        candidate_table = pd.DataFrame(
            {
                "cpdID": candidate_loadings.to_list(),
                "Loading distance": candidate_loadings_scores.to_list(),
            }
        )

        add_table(
            state,
            candidate_table,
            title="PCA candidate features",
            include_index=False,
            max_rows=50,
        )

    return {
        "message": "PCA was performed.",
        "warnings": warning_messages,
        "pca_count": state.pca_count,
        "explained_variance_percent": state.pca_per_var,
        "candidate_count": int(len(candidate_loadings)),
        "scores_path": scores_path,
        "loadings_path": loadings_path,
        "candidates_path": candidates_path,
    }


@register_operation(
    id="statistics_PLSDA",
    label="PLS-DA",
    description="Perform PLS-DA with double cross-validation and VIP-based candidate selection.",
    citation="",
    category_tags=[OperationTag.STATISTICS, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="response_column_names",
            type="str_or_list",
            required=True,
            default=None,
            label="Response column name(s)",
            help="Metadata column, or list of metadata columns, used as the response variable.",
        ),
        ParameterDef(
            name="ignored_groups",
            type="list",
            required=False,
            default=None,
            label="Ignored groups",
            help="List like [['Sample Type', 'QC'], ['Sample Type', 'Blank']].",
        ),
        ParameterDef(
            name="candidate_percentile",
            type="float",
            required=False,
            default=99.5,
            label="VIP candidate percentile",
        ),
        ParameterDef(
            name="outer_splits",
            type="int",
            required=False,
            default=5,
            label="Outer CV folds",
            help="Requested outer stratified folds; automatically capped by the smallest class.",
        ),
        ParameterDef(
            name="outer_repeats",
            type="int",
            required=False,
            default=10,
            label="Outer CV repeats",
        ),
        ParameterDef(
            name="inner_splits",
            type="int",
            required=False,
            default=5,
            label="Inner CV folds",
            help="Requested folds used to select the number of latent variables.",
        ),
        ParameterDef(
            name="selection_metric",
            type="str",
            required=False,
            default="auroc",
            label="LV selection metric",
            help="Use 'auroc' (maximize) or 'nmc' (minimize misclassifications).",
        ),
        ParameterDef(
            name="random_state",
            type="int",
            required=False,
            default=42,
            label="Random seed",
        ),
    ],
    requires=["data", "metadata"],
    produces=[
        "plsda_model",
        "plsda_stats",
        "plsda_metadata",
        "plsda_scores",
        "plsda_vip_scores",
        "figures",
        "candidates",
    ],
)
def statistics_PLSDA(
    state: WorkflowState,
    response_column_names,
    ignored_groups=None,
    candidate_percentile=99.5,
    outer_splits=5,
    outer_repeats=10,
    inner_splits=5,
    selection_metric="auroc",
    random_state=42,
):
    if state.data is None:
        raise ValueError("No data loaded in state.data.")
    if state.metadata is None:
        raise ValueError("No metadata loaded in state.metadata.")

    data = state.data.copy()
    metadata = state.metadata.copy()

    was_centered = state.was_centered
    was_scaled = state.was_scaled

    warning_messages = []

    if not was_centered:
        msg = "Data was not centered. It's highly recommended to center the data before performing PLS-DA."
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    if not was_scaled:
        msg = "Data was not scaled. It's suggested to scale the data before performing PLS-DA."
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    if "Sample File" not in metadata.columns:
        raise ValueError("Expected 'Sample File' in metadata to align samples.")

    if "cpdID" not in data.columns:
        raise ValueError("Expected a 'cpdID' feature identifier column in data.")

    if isinstance(response_column_names, str):
        response_columns = [response_column_names.strip()]
    elif isinstance(response_column_names, (list, tuple)):
        response_columns = [str(column).strip() for column in response_column_names]
    else:
        raise ValueError("response_column_names must be a column name or a list of names.")

    if not response_columns or any(not column for column in response_columns):
        raise ValueError("At least one non-empty response column name is required.")
    if len(set(response_columns)) != len(response_columns):
        raise ValueError("response_column_names contains duplicate column names.")

    missing_response_columns = [
        column for column in response_columns if column not in metadata.columns
    ]
    if missing_response_columns:
        raise ValueError(
            "Response column(s) not found in metadata: "
            + ", ".join(missing_response_columns)
        )

    if ignored_groups is None:
        ignored_groups = []
    elif (
        isinstance(ignored_groups, (list, tuple))
        and len(ignored_groups) == 2
        and not isinstance(ignored_groups[0], (list, tuple))
    ):
        ignored_groups = [ignored_groups]

    if not isinstance(ignored_groups, (list, tuple)):
        raise ValueError("ignored_groups must be a list of [column, value] pairs.")

    for pair in ignored_groups:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each ignored_groups entry must contain exactly [column, value]."
            )
        col_name, grp_name = pair
        if col_name not in metadata.columns:
            raise ValueError(
                f"Ignored-group column '{col_name}' was not found in metadata."
            )
        metadata = metadata.loc[metadata[col_name].astype(str) != str(grp_name)]

    metadata = metadata.reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No samples remain after applying ignored_groups.")

    missing_sample_mask = metadata["Sample File"].isna() | metadata[
        "Sample File"
    ].astype(str).str.strip().eq("")
    if missing_sample_mask.any():
        raise ValueError("Metadata contains an empty 'Sample File' value.")

    metadata["Sample File"] = metadata["Sample File"].astype(str)
    duplicated_samples = metadata.loc[
        metadata["Sample File"].duplicated(keep=False), "Sample File"
    ].unique()
    if len(duplicated_samples):
        raise ValueError(
            "Metadata contains duplicate 'Sample File' values: "
            + ", ".join(map(str, duplicated_samples[:10]))
        )

    data.columns = [str(column) for column in data.columns]
    sample_names = metadata["Sample File"].tolist()
    missing_samples = [sample for sample in sample_names if sample not in data.columns]
    if missing_samples:
        preview = ", ".join(missing_samples[:10])
        extra = " ..." if len(missing_samples) > 10 else ""
        raise ValueError(
            f"{len(missing_samples)} metadata sample(s) are missing from data: "
            f"{preview}{extra}"
        )

    if data["cpdID"].isna().any() or data["cpdID"].astype(str).str.strip().eq("").any():
        raise ValueError("Data contains an empty cpdID feature identifier.")
    if data["cpdID"].astype(str).duplicated().any():
        duplicates = data.loc[
            data["cpdID"].astype(str).duplicated(keep=False), "cpdID"
        ].astype(str).unique()
        raise ValueError(
            "Data contains duplicate cpdID values: " + ", ".join(duplicates[:10])
        )

    missing_response = metadata[response_columns].isna()
    blank_response = metadata[response_columns].astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    if (missing_response | blank_response).any(axis=None):
        bad_rows = metadata.loc[
            (missing_response | blank_response).any(axis=1), "Sample File"
        ].tolist()
        raise ValueError(
            "PLS-DA response values are missing for sample(s): "
            + ", ".join(map(str, bad_rows[:10]))
        )

    response_frame = metadata[response_columns].astype(str)
    response_keys = response_frame.apply(
        lambda row: json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )
    y_strat, unique_response_keys = pd.factorize(response_keys, sort=True)
    class_names = []
    for key in unique_response_keys:
        values = json.loads(key)
        if len(response_columns) == 1:
            class_names.append(str(values[0]))
        else:
            class_names.append(
                " | ".join(
                    f"{column}={value}"
                    for column, value in zip(response_columns, values)
                )
            )

    if len(class_names) < 2:
        raise ValueError(
            "PLS-DA requires at least two distinct response classes after filtering."
        )

    Y_full = np.eye(len(class_names), dtype=float)[y_strat]
    response_display = pd.Series(
        [class_names[code] for code in y_strat], index=metadata.index
    )
    metadata["PLSDA response"] = response_display

    try:
        numeric_data = data.loc[:, sample_names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "PLS-DA data contains a non-numeric abundance value. "
            "Check the selected sample columns."
        ) from exc

    X_full = numeric_data.T.to_numpy(dtype=float)
    if not np.isfinite(X_full).all():
        invalid_count = int((~np.isfinite(X_full)).sum())
        raise ValueError(
            f"PLS-DA data contains {invalid_count} missing or infinite value(s). "
            "Remove affected features or run an imputation step before PLS-DA."
        )
    if X_full.shape[1] == 0:
        raise ValueError("PLS-DA requires at least one feature.")

    try:
        candidate_percentile = float(candidate_percentile)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_percentile must be numeric.") from exc
    if not np.isfinite(candidate_percentile) or not 0 <= candidate_percentile <= 100:
        raise ValueError("candidate_percentile must be between 0 and 100.")

    outer_splits = int(outer_splits)
    outer_repeats = int(outer_repeats)
    inner_splits = int(inner_splits)
    random_state = int(random_state)
    selection_metric = str(selection_metric).strip().lower()
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must each be at least 2.")
    if outer_repeats < 1:
        raise ValueError("outer_repeats must be at least 1.")
    if selection_metric not in {"auroc", "nmc"}:
        raise ValueError("selection_metric must be 'auroc' or 'nmc'.")

    state.plsda_metadata = metadata.copy()
    class_counts = pd.Series(y_strat).value_counts().to_dict()
    min_class = min(class_counts.values()) if class_counts else 0
    if min_class < 2:
        raise ValueError(
            f"PLS-DA requires at least 2 samples in each class. Found a class with only {min_class} samples."
        )

    actual_outer_splits = min(outer_splits, min_class)

    y_cv, chosen_lvs, repeat_predictions = _plsda_double_cv_predict(
        X_full,
        Y_full,
        y_strat,
        outer_splits=actual_outer_splits,
        outer_repeats=outer_repeats,
        inner_splits=inner_splits,
        ncomp_grid=None,
        select_metric=selection_metric,
        rng=random_state,
    )
    if not chosen_lvs:
        raise RuntimeError(
            "PLS-DA cross-validation did not select any valid latent variables. "
            "No final PLS-DA model could be created. "
            "This is not an expected model result. Check the number of samples "
            "in each class, the cross-validation settings, and the input data."
        )

    lv_counts = pd.Series(chosen_lvs).value_counts().sort_index()
    modal_lvs = lv_counts.loc[lv_counts == lv_counts.max()].index
    n_comp = int(min(modal_lvs))

    lv_counts_dict = {int(k): int(v) for k, v in lv_counts.to_dict().items()}

    repeat_metrics = [
        _classification_metrics(Y_full, prediction)
        for prediction in repeat_predictions
    ]
    repeat_nmc = [metric[0] for metric in repeat_metrics]
    repeat_accuracy = [metric[1] for metric in repeat_metrics]
    repeat_auc = [metric[2] for metric in repeat_metrics]
    nmc = float(np.mean(repeat_nmc))
    cv_accuracy = float(np.mean(repeat_accuracy))
    auc = float(np.nanmean(repeat_auc))
    consensus_nmc, consensus_accuracy, consensus_auc = _classification_metrics(
        Y_full, y_cv
    )

    model = PLSRegression(n_components=n_comp, scale=False)
    model.fit(X_full, Y_full)
    calibrated_predictions = model.predict(X_full)

    vips = _vip(model)
    feature_ids = data["cpdID"].astype(str).tolist()
    state.plsda_vip_scores = pd.Series(vips, index=feature_ids, name="VIP_score")

    candidate_mask = vips > np.percentile(vips, candidate_percentile)
    candidate_vips = data.loc[candidate_mask, "cpdID"].astype(str)
    candidate_vips_scores = vips[candidate_mask]
    state.plsda_vip_candidates = candidate_vips.to_list()
    state.plsda_vip_candidates_len = len(candidate_vips)

    _add_candidates(
        state=state,
        features=candidate_vips,
        method="PLSDA-vips",
        specification=str(response_column_names),
        scores=candidate_vips_scores,
    )

    # R2 describes calibration fit of the final model.
    # Q2 describes predictive performance from held-out outer-CV predictions.

    q2_values, q2_global_flat = _q2(Y_full, repeat_predictions)
    r2_per_class = {
        str(cname): float(r2_score(Y_full[:, i], calibrated_predictions[:, i]))
        for i, cname in enumerate(class_names)
    }
    q2_per_class = {
        str(cname): float(q2_values[i])
        for i, cname in enumerate(class_names)
    }
    q2_macro = float(np.nanmean(q2_values))
    r2_macro = float(np.nanmean(list(r2_per_class.values())))
    r2_global_flat = float(
        r2_score(Y_full.ravel(), calibrated_predictions.ravel())
    )

    score_columns = [f"LV{i}" for i in range(1, model.x_scores_.shape[1] + 1)]
    scores_df = pd.DataFrame(model.x_scores_, columns=score_columns)
    scores_df.insert(0, "Sample File", sample_names)
    scores_df["PLSDA response"] = response_display.to_list()
    for response_column in response_columns:
        scores_df[response_column] = metadata[response_column].to_list()

    predicted_codes = y_cv.argmax(axis=1)
    cv_predictions_df = pd.DataFrame(
        {
            "Sample File": sample_names,
            "True class": response_display.to_list(),
            "Predicted class": [class_names[code] for code in predicted_codes],
        }
    )
    for class_index, class_name in enumerate(class_names):
        cv_predictions_df[f"CV score: {class_name}"] = y_cv[:, class_index]

    state.plsda_stats = {
        "validation_method": "Repeated double cross-validation",
        "classes": [str(c) for c in class_names],
        "class_counts": {
            str(class_names[int(code)]): int(count)
            for code, count in sorted(class_counts.items())
        },
        "outer_splits_requested": outer_splits,
        "outer_splits": actual_outer_splits,
        "outer_repeats": outer_repeats,
        "inner_splits": inner_splits,
        "select_metric": selection_metric,
        "random_state": random_state,
        "sklearn_scale": False,
        "n_components_final": n_comp,
        "LV_selection_counts": lv_counts_dict,
        "LV_selection_mode": n_comp,
        "NMC": nmc,
        "NMC_std": float(np.std(repeat_nmc, ddof=1)) if outer_repeats > 1 else 0.0,
        "NMC_per_repeat": repeat_nmc,
        "AUROC": auc,
        "AUROC_std": float(np.nanstd(repeat_auc, ddof=1)) if outer_repeats > 1 else 0.0,
        "AUROC_per_repeat": [float(value) for value in repeat_auc],
        "CV_accuracy": cv_accuracy,
        "CV_accuracy_std": (
            float(np.std(repeat_accuracy, ddof=1)) if outer_repeats > 1 else 0.0
        ),
        "CV_accuracy_per_repeat": repeat_accuracy,
        "consensus_NMC": consensus_nmc,
        "consensus_AUROC": consensus_auc,
        "consensus_CV_accuracy": consensus_accuracy,
        "R2_macro": r2_macro,
        "Q2_macro": q2_macro,
        "R2_per_class": r2_per_class,
        "Q2_per_class": q2_per_class,
        "R2_global_flat": r2_global_flat,
        "Q2_global_flat": q2_global_flat,
    }

    print("PLS-DA Double CV results:")
    print(f"  NMC (repeat mean): {nmc:.2f}")
    print(f"  AUROC (repeat mean): {auc:.4f}")
    print(f"  CV accuracy (repeat mean): {cv_accuracy:.4f}")
    print(f"  R2_macro (final-model fit): {r2_macro:.4f}")
    print(f"  Q2_macro (outer-CV prediction): {q2_macro:.4f}")
    print("")

    state.plsda_response_column = response_column_names
    state.plsda_model = model
    state.plsda = model
    state.plsda_scores = scores_df
    state.plsda_class_names = [str(name) for name in class_names]
    state.plsda_feature_ids = feature_ids
    state.plsda_score_columns = score_columns

    main_folder, statistics_folder = _ensure_statistics_folder(state)
    output_file_prefix = _get_output_file_prefix(state)

    vip_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PLSDA_VIP_scores.csv")
    )
    stats_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PLSDA_stats.csv")
    )
    candidates_path = _unique_path(
        os.path.join(
            statistics_folder, output_file_prefix + "_candidates_after_PLSDA.csv"
        )
    )
    model_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PLSDA_model.joblib")
    )
    scores_path = _unique_path(
        os.path.join(statistics_folder, output_file_prefix + "_PLSDA_scores.csv")
    )
    cv_predictions_path = _unique_path(
        os.path.join(
            statistics_folder,
            output_file_prefix + "_PLSDA_CV_predictions.csv",
        )
    )

    state.plsda_vip_scores.rename_axis("cpdID").reset_index().to_csv(
        vip_path, index=False, sep=";"
    )
    pd.DataFrame([state.plsda_stats]).to_csv(stats_path, index=False, sep=";")
    state.candidates.to_csv(candidates_path, index=False, sep=";")
    scores_df.to_csv(scores_path, index=False, sep=";")
    cv_predictions_df.to_csv(cv_predictions_path, index=False, sep=";")
    dump(
        {
            "model": model,
            "class_names": state.plsda_class_names,
            "feature_ids": feature_ids,
            "response_columns": response_columns,
            "sklearn_scale": False,
        },
        model_path,
    )

    state.plsda_model_path = model_path
    state.plsda_scores_path = scores_path
    state.plsda_vip_scores_path = vip_path
    state.plsda_cv_predictions_path = cv_predictions_path

    _add_artifact_if_available(state, vip_path, "table", "PLS-DA VIP scores")
    _add_artifact_if_available(state, stats_path, "table", "PLS-DA statistics")
    _add_artifact_if_available(state, scores_path, "table", "PLS-DA scores")
    _add_artifact_if_available(
        state, cv_predictions_path, "table", "PLS-DA cross-validated predictions"
    )
    _add_artifact_if_available(state, model_path, "model", "Fitted PLS-DA model")
    _add_artifact_if_available(
        state, candidates_path, "table", "Candidate features after PLS-DA"
    )

    # DEFAULT PLS-DA SCORE PLOT -----------------------------------------
    score_plot_path = None
    score_plot_title = None

    plt, _ = _load_plotting()

    groups = (
        scores_df["PLSDA response"]
        .astype(object)
        .where(
            scores_df["PLSDA response"].notna(),
            "Missing",
        )
    )

    # Standard 2D score plot: LV1 vs LV2
    if n_comp >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        for group in sorted(
            groups.unique(),
            key=_natural_sort_key,
        ):
            subset = scores_df.loc[groups == group]

            ax.scatter(
                subset["LV1"],
                subset["LV2"],
                label=str(group),
                alpha=0.8,
            )

        ax.axhline(
            0,
            color="grey",
            linewidth=0.7,
            alpha=0.5,
        )

        ax.axvline(
            0,
            color="grey",
            linewidth=0.7,
            alpha=0.5,
        )

        ax.set_xlabel("LV1")
        ax.set_ylabel("LV2")
        ax.set_title("PLS-DA score plot")

        score_plot_title = "PLS-DA score plot — LV1 vs LV2"

        score_plot_path = _unique_path(
            os.path.join(
                statistics_folder,
                output_file_prefix
                + "_PLSDA_scores_LV1_vs_LV2.png",
            )
        )

    # One-dimensional score plot when final model contains only LV1
    else:
        fig, ax = plt.subplots(figsize=(9, 3.5))

        rng = np.random.default_rng(random_state)

        for group in sorted(
            groups.unique(),
            key=_natural_sort_key,
        ):
            subset = scores_df.loc[groups == group]

            # Small vertical jitter is only for visual separation.
            y_jitter = rng.uniform(
                -0.08,
                0.08,
                size=len(subset),
            )

            ax.scatter(
                subset["LV1"],
                y_jitter,
                label=str(group),
                alpha=0.8,
            )

        ax.axvline(
            0,
            color="grey",
            linewidth=0.7,
            alpha=0.5,
        )

        ax.axhline(
            0,
            color="grey",
            linewidth=0.5,
            alpha=0.25,
        )

        ax.set_xlabel("LV1")

        # Y-axis has no model meaning.
        ax.set_yticks([])
        ax.set_ylabel("")

        ax.set_title("PLS-DA score plot — LV1")

        score_plot_title = "PLS-DA score plot — LV1"

        score_plot_path = _unique_path(
            os.path.join(
                statistics_folder,
                output_file_prefix
                + "_PLSDA_scores_LV1.png",
            )
        )

    ax.legend(
        title="PLS-DA response",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        score_plot_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    _add_artifact_if_available(
        state,
        score_plot_path,
        "figure",
        "PLS-DA score plot colored by model response",
    )

    # REPORTING ---------------------------------------------------------

    add_text(
        state,
        (
            f"PLS-DA was performed using the following response column(s): "
            f"{response_columns}. "
            f"The final model contains {n_comp} latent variable(s)."
        ),
        title="PLS-DA",
    )
    validation_setup_table = pd.DataFrame(
        {
            "Setting": [
                "Response column(s)",
                "Number of classes",
                "Final latent variables",
                "Outer CV folds",
                "Outer CV repeats",
                "Inner CV folds",
                "LV selection metric",
                "Random seed",
            ],
            "Value": [
                ", ".join(response_columns),
                len(class_names),
                n_comp,
                (
                    f"{actual_outer_splits} "
                    f"(requested {outer_splits})"
                ),
                outer_repeats,
                inner_splits,
                selection_metric,
                random_state,
            ],
        }
    )

    add_table(
        state,
        validation_setup_table,
        title="PLS-DA validation setup",
        include_index=False,
    )

    plsda_metrics_table = pd.DataFrame(
        {
            "Metric": [
                "Mean AUROC",
                "Mean CV accuracy",
                "Mean NMC",
                "Consensus AUROC",
                "Consensus CV accuracy",
                "Consensus NMC",
                "R2 macro",
                "Q2 macro",
            ],
            "Value": [
                f"{auc:.4f}",
                f"{cv_accuracy:.4f}",
                f"{nmc:.2f}",
                (
                    f"{consensus_auc:.4f}"
                    if np.isfinite(consensus_auc)
                    else "NA"
                ),
                f"{consensus_accuracy:.4f}",
                str(consensus_nmc),
                f"{r2_macro:.4f}",
                f"{q2_macro:.4f}",
            ],
            "SD across repeats": [
                f"{state.plsda_stats['AUROC_std']:.4f}",
                f"{state.plsda_stats['CV_accuracy_std']:.4f}",
                f"{state.plsda_stats['NMC_std']:.2f}",
                "—",
                "—",
                "—",
                "—",
                "—",
            ],
        }
    )

    add_table(
        state,
        plsda_metrics_table,
        title="PLS-DA validation performance",
        include_index=False,
    )

    add_text(
        state,
        (
            "R2 describes the goodness of fit of the final PLS-DA model fitted "
            "to the complete dataset. Q2 describes predictive performance estimated "
            "from held-out predictions in repeated outer cross-validation."
        ),
        title="R2 and Q2 interpretation",
    )
    per_class_metrics = pd.DataFrame({
        "Class": [str(name) for name in class_names],
        "R2": [r2_per_class[str(name)] for name in class_names],
        "Q2": [q2_per_class[str(name)] for name in class_names],}
    )

    lv_selection_table = pd.DataFrame(
        {
            "Latent variables": [
                int(value)
                for value in lv_counts.index
            ],
            "Times selected": [
                int(value)
                for value in lv_counts.values
            ],
        }
    )

    add_table(
        state,
        lv_selection_table,
        title="Latent-variable selection during double CV",
        include_index=False,
    )

    add_table(
        state,
        per_class_metrics,
        title="Per-class R2 and Q2",
        include_index=False,
    )
    class_count_table = pd.DataFrame({
        "Class": [str(name) for name in class_names],
        "Samples": [int(class_counts.get(i, 0)) for i in range(len(class_names))],}
    )
    add_table(
        state,
        class_count_table,
        title="PLS-DA classes",
        include_index=False,
    )

    if score_plot_path is not None:
        add_text(
            state,
            (
                "The score plot shows the first two latent variables "
                "of the final PLS-DA model fitted to the complete dataset. "
                "Samples are colored according to the response classes "
                "used to train the model."
            ),
            title="PLS-DA score plot",
        )

        add_figure(
            state,
            score_plot_path,
            title="PLS-DA score plot — LV1 vs LV2",
        )

    else:
            add_text(
        state,
        (
            "The score plot shows the latent-variable scores of the final "
            "PLS-DA model fitted to the complete dataset. "
            "Samples are colored according to the response classes used "
            "to train the model."
        ),
        title="PLS-DA score plot",
    )

    if n_comp == 1:
        add_text(
            state,
            (
                "The final model contains one latent variable. "
                "Therefore, scores are displayed along LV1 only. "
                "Small vertical jitter is used solely to prevent overlapping "
                "points and does not represent an additional model dimension."
            ),
        )

    add_figure(
        state,
        score_plot_path,
        title=score_plot_title,
    )
    
    return {
        "message": "PLS-DA was performed.",
        "warnings": warning_messages,
        "plsda_stats": state.plsda_stats,
        "vip_path": vip_path,
        "stats_path": stats_path,
        "candidates_path": candidates_path,
        "model_path": model_path,
        "scores_path": scores_path,
        "score_plot_path": score_plot_path,
        "cv_predictions_path": cv_predictions_path,
    }


@register_operation(
    id="visualizer_PLSDA",
    label="Visualize PLS-DA Scores",
    description="Create a latent-variable score plot from the fitted PLS-DA model.",
    citation="",
    category_tags=[OperationTag.STATISTICS, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="component_x",
            type="int",
            required=False,
            default=1,
            label="X latent variable",
        ),
        ParameterDef(
            name="component_y",
            type="int",
            required=False,
            default=2,
            label="Y latent variable",
        ),
        ParameterDef(
            name="color_by",
            type="str",
            required=False,
            default="PLSDA response",
            label="Color by",
            help="PLSDA response or another metadata column.",
        ),
        ParameterDef(
            name="annotate_samples",
            type="bool",
            required=False,
            default=False,
            label="Annotate samples",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
        ),
    ],
    requires=["plsda_scores"],
    produces=["figures"],
)
def visualizer_PLSDA(
    state: WorkflowState,
    component_x=1,
    component_y=2,
    color_by="PLSDA response",
    annotate_samples=False,
    plt_name_suffix="",
):
    plt, _ = _load_plotting()
    scores = state.plsda_scores.copy()
    component_x = int(component_x)
    component_y = int(component_y)
    if component_x < 1 or component_y < 1 or component_x == component_y:
        raise ValueError("Choose two different positive latent-variable numbers.")

    x_column = f"LV{component_x}"
    y_column = f"LV{component_y}"
    missing_components = [
        column for column in (x_column, y_column) if column not in scores.columns
    ]
    if missing_components:
        available = ", ".join(state.plsda_score_columns or [])
        raise ValueError(
            f"PLS-DA component(s) not available: {', '.join(missing_components)}. "
            f"Available components: {available}."
        )

    color_by = str(color_by or "").strip()
    if color_by and color_by not in scores.columns:
        metadata = getattr(state, "plsda_metadata", None)

        if metadata is None:
            metadata = state.metadata

        if metadata is None or color_by not in metadata.columns:
            raise ValueError(
                f"Color column '{color_by}' was not found in PLS-DA metadata."
            )

        if "Sample File" not in metadata.columns:
            raise ValueError(
                "Expected 'Sample File' in metadata to align PLS-DA samples."
            )

        color_values = metadata[["Sample File", color_by]].copy()
        color_values["Sample File"] = color_values["Sample File"].astype(str)

        if color_values["Sample File"].duplicated().any():
            raise ValueError(
                "Metadata contains duplicate 'Sample File' values; "
                "PLS-DA scores cannot be aligned unambiguously."
            )

        scores = scores.merge(
            color_values,
            on="Sample File",
            how="left",
            validate="one_to_one",
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_by:
        groups = scores[color_by].astype(object).where(
            scores[color_by].notna(), "Missing"
        )
        for group in sorted(groups.unique(), key=_natural_sort_key):
            subset = scores.loc[groups == group]
            ax.scatter(
                subset[x_column],
                subset[y_column],
                label=str(group),
                alpha=0.8,
            )
        ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(scores[x_column], scores[y_column], alpha=0.8)

    if annotate_samples:
        for _, row in scores.iterrows():
            ax.annotate(
                str(row["Sample File"]),
                (row[x_column], row[y_column]),
                fontsize=7,
                alpha=0.7,
            )

    ax.axhline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title("PLS-DA score plot")
    fig.tight_layout()

    _, statistics_folder = _ensure_statistics_folder(state)
    output_file_prefix = _get_output_file_prefix(state)
    suffixes = _get_suffixes(state)
    base_path = os.path.join(
        statistics_folder,
        f"{output_file_prefix}_PLSDA_scores_{x_column}_vs_{y_column}_{plt_name_suffix}",
    )
    saved_paths = []
    for suffix in suffixes:
        out_path = _unique_path(base_path + suffix)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)
    
    if saved_paths:
        _add_artifact_if_available(
            state, saved_paths[0], "figure", "PLS-DA score plot"
        )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"PLS-DA score plot created for {x_column} versus {y_column}. "
            f"Samples were colored by '{color_by}'."
            if color_by
            else
            f"PLS-DA score plot created for {x_column} versus {y_column}."
        ),
        title="PLS-DA score plot",
    )

    add_figure(
        state,
        fig,
        title=f"{x_column} vs {y_column}",
    )
    plt.close(fig)

    return {
        "message": "PLS-DA score plot was created.",
        "saved_paths": saved_paths,
        "component_x": x_column,
        "component_y": y_column,
        "color_by": color_by,
    }


@register_operation(
    id="visualizer_PLSDA_vips",
    label="Visualize PLS-DA VIP Scores",
    description="Plot the highest variable-importance-in-projection (VIP) scores.",
    citation="",
    category_tags=[OperationTag.STATISTICS, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="top_n",
            type="int",
            required=False,
            default=30,
            label="Number of features",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
        ),
    ],
    requires=["plsda_vip_scores"],
    produces=["figures"],
)
def visualizer_PLSDA_vips(state: WorkflowState, top_n=30, plt_name_suffix=""):
    plt, _ = _load_plotting()
    top_n = int(top_n)
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    vip_scores = pd.to_numeric(
        state.plsda_vip_scores, errors="coerce"
    ).dropna().sort_values(ascending=False).head(top_n)
    if vip_scores.empty:
        raise ValueError("No finite PLS-DA VIP scores are available to plot.")

    plot_values = vip_scores.sort_values()
    fig_height = max(5, 0.28 * len(plot_values))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = ["#8b4513" if value >= 1 else "#c9a27e" for value in plot_values]
    ax.barh(plot_values.index.astype(str), plot_values.values, color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="VIP = 1")
    ax.set_xlabel("VIP score")
    ax.set_ylabel("cpdID")
    ax.set_title(f"Top {len(plot_values)} PLS-DA VIP scores")
    ax.legend(frameon=False)
    fig.tight_layout()

    _, statistics_folder = _ensure_statistics_folder(state)
    output_file_prefix = _get_output_file_prefix(state)
    suffixes = _get_suffixes(state)
    base_path = os.path.join(
        statistics_folder,
        f"{output_file_prefix}_PLSDA_VIP_top_{top_n}_{plt_name_suffix}",
    )
    saved_paths = []
    for suffix in suffixes:
        out_path = _unique_path(base_path + suffix)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)

    if saved_paths:
        _add_artifact_if_available(
            state, saved_paths[0], "figure", "PLS-DA VIP score plot"
        )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"The {len(plot_values)} highest PLS-DA VIP scores are shown below. "
            f"The dashed reference line indicates VIP = 1."
        ),
        title="PLS-DA VIP scores",
    )

    add_figure(
        state,
        fig,
        title=f"Top {len(plot_values)} PLS-DA VIP scores",
    )

    plt.close(fig)
    return {
        "message": "PLS-DA VIP score plot was created.",
        "saved_paths": saved_paths,
        "feature_count": int(len(plot_values)),
    }


@register_operation(
    id="statistics_ttest",
    label="t-test / Mann-Whitney U",
    description="Compare two selected groups feature-by-feature using t-test or Mann-Whitney U test based on Shapiro normality checks.",
    citation="",
    category_tags=[OperationTag.STATISTICS],
    parameter_schema=[
        ParameterDef(
            name="groups_column_name",
            type="str",
            required=True,
            default=None,
            label="Grouping metadata column",
            help="Metadata column containing the compared groups.",
        ),
        ParameterDef(
            name="group1",
            type="str",
            required=True,
            default=None,
            label="Group 1",
        ),
        ParameterDef(
            name="group2",
            type="str",
            required=True,
            default=None,
            label="Group 2",
        ),
        ParameterDef(
            name="p_value_correction_method",
            type="str",
            required=False,
            default=None,
            label="P-value correction method",
            help="Use 'fdr_bh' (Benjamini-Hochberg), 'bonferroni', or leave empty for none.",
        ),
        ParameterDef(
            name="table_name_suffix",
            type="str",
            required=False,
            default="ttest_results",
            label="Table name suffix",
        ),
    ],
    requires=["data", "metadata"],
    produces=["fold_change"],
)
def statistics_ttest(
    state: WorkflowState,
    groups_column_name,
    group1,
    group2,
    p_value_correction_method=None,
    table_name_suffix="ttest_results",
):
    if state.data is None:
        raise ValueError("No data loaded in state.data.")
    if state.metadata is None:
        raise ValueError("No metadata loaded in state.metadata.")

    data = state.data.copy()
    metadata = state.metadata.copy()
    output_file_prefix = _get_output_file_prefix(state)
    was_centered = state.was_centered
    was_scaled = state.was_scaled
    was_log_transformed = state.was_log_transformed
    log_base = state.log_base

    warning_messages = []

    if was_centered:
        msg = "Running t-test on centered data; interpretation may be affected. Consider computing fold change prior to centering or scaling."
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    if was_scaled:
        raise ValueError(
            "A t-test cannot be run on scaled data. Use unscaled data because "
            "scaling changes feature variances and invalidates the test."
        )

    if groups_column_name not in metadata.columns:
        raise ValueError(f"Column '{groups_column_name}' was not found in metadata.")
    if "Sample File" not in metadata.columns:
        raise ValueError("Expected 'Sample File' in metadata to align samples.")
    if "cpdID" not in data.columns:
        raise ValueError("Expected a 'cpdID' feature identifier column in data.")

    normalized_groups = metadata[groups_column_name].astype(str)
    group1 = str(group1)
    group2 = str(group2)
    available_groups = normalized_groups.loc[
        metadata[groups_column_name].notna()
    ].unique().tolist()
    if group1 not in available_groups:
        raise ValueError(f"Group '{group1}' was not found in '{groups_column_name}'.")
    if group2 not in available_groups:
        raise ValueError(f"Group '{group2}' was not found in '{groups_column_name}'.")
    if group1 == group2:
        raise ValueError("group1 and group2 must be different groups.")

    group1_samples = metadata.loc[
        normalized_groups == group1, "Sample File"
    ].astype(str).tolist()
    group2_samples = metadata.loc[
        normalized_groups == group2, "Sample File"
    ].astype(str).tolist()
    if len(group1_samples) < 2 or len(group2_samples) < 2:
        raise ValueError("Each compared group must contain at least two samples.")

    data.columns = [str(column) for column in data.columns]
    missing_samples = [
        sample
        for sample in group1_samples + group2_samples
        if sample not in data.columns
    ]
    if missing_samples:
        raise ValueError(
            "Metadata sample(s) missing from data: "
            + ", ".join(dict.fromkeys(missing_samples))
        )

    try:
        group1_data = data[group1_samples].apply(pd.to_numeric, errors="raise")
        group2_data = data[group2_samples].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Statistical test input contains non-numeric values.") from exc

    g1_mean = group1_data.mean(axis=1)
    g2_mean = group2_data.mean(axis=1)
    if was_log_transformed:
        if log_base is None or float(log_base) <= 0 or float(log_base) == 1:
            raise ValueError(
                "A valid log_base is required for log-transformed data."
            )

        # Difference between means in log space corresponds to fold change
        # in the original scale.
        fold_change = float(log_base) ** (g2_mean - g1_mean)

    elif was_centered:
        # Ratios cannot be reconstructed from centered non-log data.
        fold_change = pd.Series(
            np.nan,
            index=data.index,
            dtype=float,
        )

        msg = (
            "Fold change was not calculated because the data are centered "
            "but not log-transformed. The original group means are required "
            "for a meaningful ratio."
        )
        warnings.warn(msg, UserWarning)
        warning_messages.append(msg)

    else:
        # Do not invent a pseudocount. A zero denominator means that the
        # fold-change ratio is undefined.
        fold_change = g2_mean.div(
            g1_mean.replace(0, np.nan)
        )

    normality_group1 = []
    normality_group2 = []

    for i in range(len(data)):
        vals1 = group1_data.iloc[i, :].dropna().to_numpy()
        vals2 = group2_data.iloc[i, :].dropna().to_numpy()

        normality_group1.append(_shapiro_ok(vals1))
        normality_group2.append(_shapiro_ok(vals2))

    p_values = []
    tests_used = []
    for i in range(len(data)):
        vals1 = group1_data.iloc[i, :].dropna().to_numpy()
        vals2 = group2_data.iloc[i, :].dropna().to_numpy()

        ng1 = normality_group1[i]
        ng2 = normality_group2[i]

        if ng1 == True and ng2 == True:
            stat, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy="omit")
            tests_used.append("t-test")
        else:
            try:
                stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
                tests_used.append("Mann-Whitney U")
            except ValueError:
                p = np.nan
                tests_used.append("not calculated")

        p_values.append(p)
    p_values = pd.Series(p_values)
    adjusted_p_values, correction_method = _adjust_pvalues(
        p_values, p_value_correction_method
    )

    both_normal = []
    for g1, g2 in zip(normality_group1, normality_group2):
        if (
            g1 == "Not enough data for normality test"
            or g2 == "Not enough data for normality test"
        ):
            both_normal.append("Not enough data for normality test")
        else:
            both_normal.append(g1 and g2)

    p_values_table = pd.DataFrame(
        {
            "cpdID": data["cpdID"],
            "Fold Change": fold_change,
            "p-value": p_values,
            "adjusted p-value": adjusted_p_values,
            "both groups normal": both_normal,
            "used test": tests_used,
            "group": [f"{group1} vs {group2}"] * len(data),
        }
    )

    sort_column = (
        "adjusted p-value" if correction_method != "none" else "p-value"
    )
    p_values_table = p_values_table.sort_values(
        by=sort_column, ascending=True
    ).reset_index(drop=True)

    state.fold_change = p_values_table
    state.fold_change_count += 1

    main_folder, statistics_folder = _ensure_statistics_folder(state)
    csv_name = _unique_path(
        os.path.join(
            statistics_folder,
            output_file_prefix
            + "-"
            + str(group1)
            + "_vs_"
            + str(group2)
            + str(table_name_suffix)
            + ".csv",
        )
    )
    p_values_table.to_csv(csv_name, index=False, sep=";")
    _add_artifact_if_available(
        state, csv_name, "table", "t-test / Mann-Whitney results"
    )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Feature-by-feature statistical comparison was performed between "
            f"'{group1}' (n={len(group1_samples)}) and "
            f"'{group2}' (n={len(group2_samples)}). "
            f"Grouping was based on metadata column '{groups_column_name}'."
        ),
        title="t-test / Mann-Whitney U",
    )
    add_text(
        state,
        (
            f"Welch's t-test was used when both groups passed the normality check; "
            f"otherwise the Mann-Whitney U test was used. "
            f"P-value correction method: {correction_method}. "
            f"Fold change is expressed as {group2} / {group1}."
        ),
        title="Statistical testing",
    )
    add_table(
        state,
        p_values_table,
        title="Statistical results",
        include_index=False,
        max_rows=100,
    )

    add_text(
        state,
        f"Full statistical results were saved to: {csv_name}",
        title="Saved results",
    )

    return {
        "message": "t-test / Mann-Whitney U was performed.",
        "warnings": warning_messages,
        "group1": group1,
        "group2": group2,
        "group1_n": len(group1_samples),
        "group2_n": len(group2_samples),
        "p_value_correction_method": correction_method,
        "table_path": csv_name,
    }


@register_operation(
    id="visualize_PCA_scores",
    label="Visualize PCA Scores",
    description="Create a PCA score plot from previously calculated PCA results.",
    citation="",
    category_tags=[OperationTag.STATISTICS, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="component_x",
            type="int",
            required=False,
            default=1,
            label="X component",
            help="Principal component shown on the x-axis.",
        ),
        ParameterDef(
            name="component_y",
            type="int",
            required=False,
            default=2,
            label="Y component",
            help="Principal component shown on the y-axis.",
        ),
        ParameterDef(
            name="color_by",
            type="str",
            required=False,
            default="",
            label="Color by metadata column",
            help="Optional metadata column used for coloring samples, e.g. Sample Type or Diagnosis.",
        ),
        ParameterDef(
            name="annotate_samples",
            type="bool",
            required=False,
            default=False,
            label="Annotate samples",
            help="If True, sample names are written next to points.",
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
    requires=["pca_df", "pca_per_var"],
    produces=["figures"],
)


def visualize_PCA_scores(
    state: WorkflowState,
    component_x=1,
    component_y=2,
    color_by="",
    annotate_samples=False,
    plt_name_suffix="",
):
    plt, sns = _load_plotting()

    if state.pca_df is None:
        raise ValueError("PCA results not found. Run PCA first.")

    if state.pca_per_var is None:
        raise ValueError("PCA explained variance not found. Run PCA first.")

    if state.data is None:
        raise ValueError("No data found in state.data.")

    pca_df = state.pca_df.copy()

    component_x = int(component_x)
    component_y = int(component_y)

    pcx = "PC" + str(component_x)
    pcy = "PC" + str(component_y)

    if pcx not in pca_df.columns:
        raise ValueError(f"{pcx} not found in PCA results.")

    if pcy not in pca_df.columns:
        raise ValueError(f"{pcy} not found in PCA results.")

    sample_names = list(state.data.columns[1:])

    if len(sample_names) != len(pca_df):
        raise ValueError(
            "Number of PCA score rows does not match number of sample columns."
        )

    pca_df["Sample File"] = sample_names

    fig, ax = plt.subplots(figsize=(8, 6))

    color_by = "" if color_by is None else str(color_by).strip()

    if color_by:
        if state.metadata is None:
            raise ValueError(
                f"Cannot color PCA scores by '{color_by}' because metadata are missing."
            )

        if color_by not in state.metadata.columns:
            raise ValueError(
                f"Color column '{color_by}' was not found in metadata."
            )

        metadata = state.metadata.copy()

        if "Sample File" not in metadata.columns:
            raise ValueError("Expected 'Sample File' column in metadata.")

        metadata = metadata[["Sample File", color_by]].copy()
        metadata["Sample File"] = metadata["Sample File"].astype(str)

        if metadata["Sample File"].duplicated().any():
            duplicates = metadata.loc[
                metadata["Sample File"].duplicated(keep=False),
                "Sample File",
            ].unique()

            raise ValueError(
                "Metadata contains duplicate 'Sample File' values: "
                + ", ".join(map(str, duplicates[:10]))
            )

        pca_df = pca_df.merge(
            metadata,
            on="Sample File",
            how="left",
            validate="one_to_one",
        )

        group_values = pca_df[color_by].astype(object).where(pca_df[color_by].notna(),"Missing",)

        groups = group_values.unique()

        for group in groups:
            subset = pca_df.loc[group_values == group]

            ax.scatter(
                subset[pcx],
                subset[pcy],
                label=str(group),
                alpha=0.8,
            )

        ax.legend(
            title=color_by,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
        )

    else:
        ax.scatter(
            pca_df[pcx],
            pca_df[pcy],
            alpha=0.8,
        )

    if annotate_samples:
        for _, row in pca_df.iterrows():
            ax.text(
                row[pcx],
                row[pcy],
                str(row["Sample File"]),
                fontsize=7,
                alpha=0.7,
            )

    per_var = state.pca_per_var

    ax.set_xlabel(f"{pcx} - {per_var[component_x - 1]}%")
    ax.set_ylabel(f"{pcy} - {per_var[component_y - 1]}%")
    ax.set_title("PCA score plot")
    ax.axhline(0, color="grey", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.7, alpha=0.5)

    fig.tight_layout()

    main_folder, statistics_folder = _ensure_statistics_folder(state)
    suffixes = _get_suffixes(state)
    output_file_prefix = _get_output_file_prefix(state)

    base_name = (
        output_file_prefix
        + "_PCA_scores_"
        + pcx
        + "_vs_"
        + pcy
        + "_"
        + str(plt_name_suffix)
    )

    base_path = os.path.join(statistics_folder, base_name)

    saved_paths = []

    for suffix in suffixes:
        out_path = _unique_path(base_path + suffix)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(out_path)

    if len(saved_paths) > 0:
        figure_base_path = os.path.splitext(saved_paths[0])[0]
    else:
        figure_base_path = base_path

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"PCA score plot created for {pcx} versus {pcy}. "
            f"{pcx} explains {per_var[component_x - 1]}% and "
            f"{pcy} explains {per_var[component_y - 1]}% of the variance."
        ),
        title="PCA score plot",
    )
    if color_by:
        add_text(
            state,
            f"Samples were colored according to metadata column '{color_by}'.",
        )
    add_figure(
        state,
        fig,
        title=f"{pcx} vs {pcy}",
    )

    _add_artifact_if_available(
        state,
        saved_paths[0] if saved_paths else figure_base_path,
        "figure",
        "PCA score plot",
    )

    plt.close(fig)
    return {
        "message": "PCA score plot was created.",
        "figure_base_path": figure_base_path,
        "saved_paths": saved_paths,
        "component_x": pcx,
        "component_y": pcy,
        "color_by": color_by,
    }


@register_operation(
    id="visualizer_PCA_grouped",
    label="Visualize PCA Grouped",
    description="Create a PCA plot with colors based on one metadata column and markers based on another metadata column.",
    citation="",
    category_tags=[OperationTag.STATISTICS, OperationTag.VISUALIZATION],
    parameter_schema=[
        ParameterDef(
            name="color_column",
            type="str_or_none",
            required=False,
            default=None,
            label="Color column",
            help="Metadata column used for colors. Use None if no color grouping should be used.",
        ),
        ParameterDef(
            name="marker_column",
            type="str_or_none",
            required=False,
            default=None,
            label="Marker column",
            help="Metadata column used for marker shapes. Use None if no marker grouping should be used.",
        ),
        ParameterDef(
            name="cmap",
            type="str",
            required=False,
            default="nipy_spectral",
            label="Colormap",
            help="Matplotlib colormap name, e.g. nipy_spectral, viridis, plasma, turbo.",
        ),
        ParameterDef(
            name="crossout_outliers",
            type="bool",
            required=False,
            default=False,
            label="Cross out outliers",
            help="If True, PCA outliers with absolute PC1 or PC2 z-score > 3 are marked with an X.",
        ),
        ParameterDef(
            name="zoom_in_group",
            type="str_or_none",
            required=False,
            default=None,
            label="Zoom in group",
            help="If specified, zooms the plot to this group value.",
        ),
        ParameterDef(
            name="plt_name_suffix",
            type="str",
            required=False,
            default="",
            label="Plot name suffix",
            help="Suffix added to the saved plot filename.",
        ),
        ParameterDef(
            name="graph_title",
            type="str",
            required=False,
            default="PCA graph",
            label="Graph title",
            help="Title shown above the PCA plot.",
        ),
        ParameterDef(
            name="annotate_samples",
            type="bool",
            required=False,
            default=False,
            label="Annotate samples",
            help="If True, the farthest samples from the origin are annotated.",
        ),
        ParameterDef(
            name="nm_to_annotate",
            type="int",
            required=False,
            default=10,
            label="Number to annotate",
            help="Number of farthest samples from the origin to annotate.",
        ),
        ParameterDef(
            name="ignore_nans_in_groups",
            type="bool",
            required=False,
            default=True,
            label="Ignore NaNs in groups",
            help="If True, samples with NaN in grouping columns are ignored.",
        ),
    ],
    requires=["pca_df", "pca_per_var", "metadata"],
    produces=["figures"],
)
def visualizer_PCA_grouped(
    state: WorkflowState,
    color_column=None,
    marker_column=None,
    cmap="nipy_spectral",
    crossout_outliers=False,
    zoom_in_group=None,
    plt_name_suffix="",
    graph_title="PCA graph",
    annotate_samples=False,
    nm_to_annotate=10,
    ignore_nans_in_groups=True,
):
    """
    Reworked version for PySPRESSO-APP
    """

    plt, sns = _load_plotting()

    import matplotlib as mpl
    from matplotlib.patches import Ellipse
    from adjustText import adjust_text

    # GUI may pass "None" as a string, so normalize it to real None.
    if color_column in ["", "None", "none", "null", "NULL"]:
        color_column = None

    if marker_column in ["", "None", "none", "null", "NULL"]:
        marker_column = None

    if zoom_in_group in ["", "None", "none", "null", "NULL"]:
        zoom_in_group = None

    metadata = state.metadata.copy()
    original_metadata = state.metadata.copy()

    output_file_prefix = _get_output_file_prefix(state)
    main_folder, statistics_folder = _ensure_statistics_folder(state)
    suffixes = _get_suffixes(state)

    if state.pca_data is None:
        raise ValueError("PCA was not performed yet. Run PCA first.")

    if state.pca_df is None:
        raise ValueError("PCA score table was not found. Run PCA first.")

    if state.pca_per_var is None:
        raise ValueError("PCA explained variance was not found. Run PCA first.")

    pca_df = state.pca_df.copy()
    per_var = state.pca_per_var

    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        raise ValueError("PCA score table must contain PC1 and PC2 columns.")

    if color_column is not None and color_column not in metadata.columns:
        raise ValueError(f"Color column '{color_column}' was not found in metadata.")

    if marker_column is not None and marker_column not in metadata.columns:
        raise ValueError(f"Marker column '{marker_column}' was not found in metadata.")

    column_name = color_column
    second_column_name = marker_column

    # Handle NaNs in grouping columns, following the original function.
    if ignore_nans_in_groups:
        cols = [c for c in [column_name, second_column_name] if c is not None]

        if len(cols) > 0:
            mask = metadata[cols].notna().all(axis=1)
            metadata = metadata.loc[mask]
            pca_df = pca_df.loc[mask]
    else:
        for col in (column_name, second_column_name):
            if col is not None:
                metadata[col] = (
                    metadata[col].astype(object).where(metadata[col].notna(), "nan")
                )

    cmap = mpl.colormaps[cmap]

    if crossout_outliers:
        pca_df["PC1_zscore"] = zscore(pca_df["PC1"])
        pca_df["PC2_zscore"] = zscore(pca_df["PC2"])

        outliers = pca_df[
            (np.abs(pca_df["PC1_zscore"]) > 3) | (np.abs(pca_df["PC2_zscore"]) > 3)
        ]

    # Get colors for the first column.
    if column_name is not None:
        column_unique_values = metadata[column_name].unique()
        column_unique_values = sorted(column_unique_values, key=_natural_sort_key)

        num_unique_values = len(column_unique_values)
        color_indices = np.linspace(0.05, 0.95, num_unique_values)
        colors = [cmap(i) for i in color_indices]
        class_type_colors = dict(zip(column_unique_values, colors))

    # Get markers for the second column.
    if second_column_name is not None:
        second_column_unique_values = metadata[second_column_name].unique()
        second_column_unique_values = sorted(
            second_column_unique_values,
            key=_natural_sort_key,
        )

        markers = cycle(
            ["o", "v", "s", "^", "*", "<", "p", ">", "h", "H", "D", "d", "P", "X"]
        )
        class_type_markers = dict(zip(second_column_unique_values, markers))

    if column_name is not None and second_column_name is not None:
        existing_combinations = set(
            zip(metadata[column_name], metadata[second_column_name])
        )
        existing_combinations = sorted(existing_combinations, key=_natural_sort_key)

    elif column_name is not None:
        existing_combinations = set(metadata[column_name])
        existing_combinations = sorted(existing_combinations, key=_natural_sort_key)

    elif second_column_name is not None:
        existing_combinations = set(metadata[second_column_name])
        existing_combinations = sorted(existing_combinations, key=_natural_sort_key)

    else:
        raise ValueError(
            "At least one of the columns must be specified. "
            "Either color_column or marker_column or both."
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Iterate over unique combinations of the two columns.
    for combination in existing_combinations:
        if isinstance(combination, tuple):
            sample_type, class_type = combination
        else:
            if column_name is not None:
                sample_type = combination
                class_type = None
            else:
                sample_type = None
                class_type = combination

        if sample_type is not None and class_type is not None:
            df_samples = pca_df.loc[
                (metadata[column_name] == sample_type)
                & (metadata[second_column_name] == class_type)
            ]
        elif sample_type is not None:
            df_samples = pca_df.loc[metadata[column_name] == sample_type]
        elif class_type is not None:
            df_samples = pca_df.loc[metadata[second_column_name] == class_type]

        color = (
            "grey"
            if sample_type is None
            else class_type_colors.get(sample_type, "grey")
        )

        marker = "o" if class_type is None else class_type_markers.get(class_type, "o")

        label = f"{sample_type or ''} - {class_type or ''}".strip(" -")

        plt.scatter(
            df_samples["PC1"],
            df_samples["PC2"],
            color=color,
            marker=marker,
            label=label,
            alpha=0.6,
            s=20,
        )

        # Zoom in to specific group if specified.
        if zoom_in_group is not None:
            if (
                column_name is not None
                and zoom_in_group in metadata[column_name].values
            ):
                df_zoom = pca_df.loc[metadata[column_name] == zoom_in_group]
                plt.xlim(df_zoom["PC1"].min() - 0.5, df_zoom["PC1"].max() + 0.5)
                plt.ylim(df_zoom["PC2"].min() - 0.5, df_zoom["PC2"].max() + 0.5)

            elif (
                second_column_name is not None
                and zoom_in_group in metadata[second_column_name].values
            ):
                df_zoom = pca_df.loc[metadata[second_column_name] == zoom_in_group]
                plt.xlim(df_zoom["PC1"].min() - 0.5, df_zoom["PC1"].max() + 0.5)
                plt.ylim(df_zoom["PC2"].min() - 0.5, df_zoom["PC2"].max() + 0.5)

        # Draw ellipse around group if enough samples are present.
        if len(df_samples) >= 3:
            covmat = np.cov(df_samples[["PC1", "PC2"]].values.T)

            if np.isinf(covmat).any() or np.isnan(covmat).any():
                print(
                    f"Skipping ellipse for combination {combination} "
                    "due to invalid covariance matrix."
                )
                continue

            lambda_, v = np.linalg.eig(covmat)
            lambda_ = np.sqrt(lambda_)

            ell = Ellipse(
                xy=(np.mean(df_samples["PC1"]), np.mean(df_samples["PC2"])),
                width=lambda_[0] * 2,
                height=lambda_[1] * 2,
                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                edgecolor=color,
                lw=1,
                facecolor="none",
                alpha=0.6,
            )
            plt.gca().add_artist(ell)

    texts = []

    if annotate_samples:
        pca_df["distance"] = np.linalg.norm(pca_df[["PC1", "PC2"]].values, axis=1)
        farthest = pca_df.nlargest(int(nm_to_annotate), "distance")

        for i, row in farthest.iterrows():
            sample_name = original_metadata.iloc[i]["Sample File"]

            texts.append(
                ax.text(
                    row["PC1"],
                    row["PC2"],
                    sample_name,
                    fontsize=8,
                    color="black",
                    alpha=0.7,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                )
            )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    )

    if crossout_outliers:
        plt.scatter(
            outliers["PC1"],
            outliers["PC2"],
            color="black",
            marker="x",
            label="Outliers",
            alpha=0.6,
            s=20,
        )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    plt.title(graph_title)
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC2 - {0}%".format(per_var[1]))

    plt_name = os.path.join(
        statistics_folder,
        output_file_prefix + "_PCA_grouped_" + str(plt_name_suffix),
    )

    saved_paths = []

    for suffix in suffixes:
        out_path = _unique_path(plt_name + suffix)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        saved_paths.append(out_path)

    if len(saved_paths) > 0:
        figure_base_path = os.path.splitext(saved_paths[0])[0]
    else:
        figure_base_path = plt_name

    # REPORTING ---------------------------------------------------------
    grouping_description = []

    if color_column is not None:
        grouping_description.append(
            f"colors represent '{color_column}'"
        )

    if marker_column is not None:
        grouping_description.append(
            f"marker shapes represent '{marker_column}'"
        )

    add_text(
        state,
        (
            f"Grouped PCA score plot was created; "
            f"{' and '.join(grouping_description)}. "
            f"Outlier crossing enabled: {crossout_outliers}. "
            f"Zoomed group: {zoom_in_group if zoom_in_group is not None else 'none'}."
        ),
        title="Grouped PCA plot",
    )

    add_figure(
        state,
        fig,
        title=graph_title,
    )

    plt.close(fig)
    return {
        "figure_base_path": figure_base_path,
        "saved_paths": saved_paths,
        "color_column": color_column,
        "marker_column": marker_column,
        "crossout_outliers": crossout_outliers,
        "zoom_in_group": zoom_in_group,
        "annotate_samples": annotate_samples,
    }
