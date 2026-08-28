# pyspresso_app/operations/statistics_ops.py

import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState

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
    if not hasattr(state, "artifacts"):
        return

    state.artifacts.append(
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

    names = None
    formulas = None
    annotdeltamass = None
    annotation_mw = None

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

    if variable_metadata is not None and "cpdID" in variable_metadata.columns:
        vm_cpd = variable_metadata["cpdID"].astype(str)

        if "Name" in variable_metadata.columns:
            names = variable_metadata[vm_cpd.isin(features)]["Name"].tolist()
        elif "Compound Name" in variable_metadata.columns:
            names = variable_metadata[vm_cpd.isin(features)]["Compound Name"].tolist()
        if "Formula" in variable_metadata.columns:
            formulas = variable_metadata[vm_cpd.isin(features)]["Formula"].tolist()
        if "Annot. DeltaMass [ppm]" in variable_metadata.columns:
            annotdeltamass = variable_metadata[vm_cpd.isin(features)][
                "Annot. DeltaMass [ppm]"
            ].tolist()
        if "Annotation MW" in variable_metadata.columns:
            annotation_mw = variable_metadata[vm_cpd.isin(features)][
                "Annotation MW"
            ].tolist()

    if (
        names is not None
        and formulas is not None
        and annotdeltamass is not None
        and annotation_mw is not None
    ):
        new_candidates = pd.DataFrame(
            {
                "feature": features,
                "method": method,
                "specification": specification,
                "score": scores,
                "hits": hits,
                "Name": names,
                "Formula": formulas,
                "Annot. DeltaMass [ppm]": annotdeltamass,
                "Annotation MW": annotation_mw,
            }
        )
    elif names is not None:
        new_candidates = pd.DataFrame(
            {
                "feature": features,
                "method": method,
                "specification": specification,
                "score": scores,
                "hits": hits,
                "Name": names,
            }
        )
    else:
        new_candidates = pd.DataFrame(
            {
                "feature": features,
                "method": method,
                "specification": specification,
                "score": scores,
                "hits": hits,
            }
        )

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

                        m = PLSRegression(n_components=int(lv))
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

            m_outer = PLSRegression(n_components=int(best_lv))
            m_outer.fit(X_tr, Y_tr)
            y_hat = m_outer.predict(X_te)

            pred_sum[te_outer] += y_hat
            pred_count[te_outer] += 1

    y_cv_outer = pred_sum / np.maximum(pred_count[:, None], 1)

    return y_cv_outer, chosen_lvs


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
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips


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
    report = state.report
    output_file_prefix = _get_output_file_prefix(state)
    main_folder, statistics_folder = _ensure_statistics_folder(state)
    suffixes = _get_suffixes(state)

    if data is None:
        raise ValueError("No data loaded in state.data.")
    if metadata is None:
        raise ValueError("No metadata loaded in state.metadata.")
    if column_name not in metadata.columns:
        raise ValueError(f"Column '{column_name}' was not found in metadata.")

    if min_max[0] < -1:
        min_max[0] = -1
    if min_max[1] > 1:
        min_max[1] = 1
    if min_max[0] > min_max[1]:
        min_max = [-1, 1]

    # Kept intentionally equivalent to the old PySPRESSO implementation.
    grouped_means = data.iloc[:, 1:].groupby(metadata[column_name]).mean()

    # Kept intentionally equivalent to the old PySPRESSO implementation.
    # The old code accepted 'method' but did not pass it into corr().
    correlation_matrix = grouped_means.T.corr()

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

    if report is not None:
        text = (
            "Group correlation matrix heatmap was created. Grouping is based on: "
            + column_name
        )
        report.add_together([("text", text), ("image", image_for_report), "pagebreak"])

    return {
        "message": "Group correlation matrix heatmap was created.",
        "column_name": column_name,
        "method_parameter": method,
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
    report = state.report

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

    if report is not None:
        report.add_text("<b>PCA (" + str(state.pca_count) + ") was performed. </b>")

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
    category_tags=[OperationTag.STATISTICS],
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
    ],
    requires=["data", "metadata"],
    produces=[
        "plsda_model",
        "plsda_stats",
        "plsda_metadata",
        "plsda_vip_scores",
        "candidates",
    ],
)
def statistics_PLSDA(
    state: WorkflowState,
    response_column_names,
    ignored_groups=None,
    candidate_percentile=99.5,
):
    data = state.data.copy()
    metadata = state.metadata.copy()
    report = state.report

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

    n_comp = 2
    n_folds = 10
    rng = 42

    if ignored_groups is None:
        ignored_groups = []
    if (
        isinstance(ignored_groups, list)
        and len(ignored_groups) > 0
        and isinstance(ignored_groups[0], str)
    ):
        ignored_groups = [ignored_groups]

    for col_name, grp_name in ignored_groups:
        metadata = metadata[metadata[col_name] != grp_name]
    metadata = metadata.reset_index(drop=True)

    if "Sample File" not in metadata.columns:
        raise ValueError("Expected 'Sample File' in metadata to align samples.")

    columns_to_keep = ["cpdID"] + metadata["Sample File"].tolist()
    data = data.loc[:, data.columns.isin(columns_to_keep)]

    fixed_cols = ["cpdID"]
    sample_cols = [c for c in metadata["Sample File"].tolist() if c in data.columns]
    data = pd.concat([data[fixed_cols], data[sample_cols]], axis=1)

    if isinstance(response_column_names, list):
        tmp_col = str(response_column_names)
        metadata[tmp_col] = metadata[response_column_names].apply(
            lambda x: "_".join(x.map(str)), axis=1
        )
        response_col = tmp_col
    else:
        response_col = str(response_column_names)

    state.plsda_metadata = metadata.copy()
    y_labels = pd.Categorical(metadata[response_col])
    y_strat = y_labels.codes
    y_onehot = pd.get_dummies(y_labels)
    class_names = list(y_onehot.columns)

    X_full = data.iloc[:, 1:].T.values
    Y_full = y_onehot.values

    class_counts = pd.Series(y_strat).value_counts().to_dict()
    min_class = min(class_counts.values()) if len(class_counts) else 0
    if min_class < 2:
        raise ValueError(
            f"PLS-DA requires at least 2 samples in each class. Found a class with only {min_class} samples."
        )
    if min_class < n_folds:
        new_folds = max(2, min_class)
        if new_folds != n_folds:
            n_folds = new_folds

    outer_splits = 5
    outer_repeats = 10
    inner_splits = 5
    select_metric = "auroc"

    if min_class < outer_splits:
        outer_splits = max(2, min_class)

    y_cv, chosen_lvs = _plsda_double_cv_predict(
        X_full,
        Y_full,
        y_strat,
        outer_splits=outer_splits,
        outer_repeats=outer_repeats,
        inner_splits=inner_splits,
        ncomp_grid=None,
        select_metric=select_metric,
        rng=rng,
    )

    lv_counts = pd.Series(chosen_lvs).value_counts()
    n_comp = int(lv_counts.index[0])

    lv_counts_dict = {int(k): int(v) for k, v in lv_counts.to_dict().items()}
    lv_mode = n_comp

    y_true_int = Y_full.argmax(axis=1)
    y_pred_int = y_cv.argmax(axis=1)
    nmc = int((y_true_int != y_pred_int).sum())
    cv_accuracy = float((y_true_int == y_pred_int).mean())

    try:
        if Y_full.shape[1] == 2:
            auc = float(roc_auc_score(Y_full[:, 1], y_cv[:, 1]))
        else:
            auc = float(roc_auc_score(Y_full, y_cv, multi_class="ovr", average="macro"))
    except ValueError as exc:
        warnings.warn(f"AUROC could not be computed: {exc}", UserWarning)
        auc = np.nan

    model = PLSRegression(n_components=n_comp)
    model.fit(X_full, Y_full)

    vips = _vip(model)
    state.plsda_vip_scores = pd.Series(vips, index=data.iloc[:, 0])

    candidate_mask = vips > np.percentile(vips, candidate_percentile)
    candidate_vips = data.iloc[:, 0][candidate_mask]
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

    r2_per_class = {}
    q2_per_class = {}
    for i, cname in enumerate(class_names):
        yt = Y_full[:, i]
        yp = y_cv[:, i]
        r2_i = r2_score(yt, yp)
        mse_i = mean_squared_error(yt, yp)
        var_i = np.var(yt)
        q2_i = 1.0 - (mse_i / var_i) if var_i > 0 else np.nan
        r2_per_class[cname] = float(r2_i)
        q2_per_class[cname] = float(q2_i)

    q2_macro = float(np.nanmean(list(q2_per_class.values())))
    r2_macro = float(np.nanmean(list(r2_per_class.values())))

    r2_global_flat = float(r2_score(Y_full.ravel(), y_cv.ravel()))
    mse_global = float(mean_squared_error(Y_full.ravel(), y_cv.ravel()))
    var_global = float(np.var(Y_full.ravel()))
    q2_global_flat = (
        float(1.0 - (mse_global / var_global)) if var_global > 0 else np.nan
    )

    state.plsda_stats = {
        "validation_method": "Double CV",
        "n_folds": n_folds,
        "classes": [str(c) for c in class_names],
        "outer_splits": outer_splits,
        "outer_repeats": outer_repeats,
        "inner_splits": inner_splits,
        "select_metric": select_metric,
        "n_components_final": n_comp,
        "LV_selection_counts": lv_counts_dict,
        "LV_selection_mode": lv_mode,
        "NMC": nmc,
        "AUROC": auc,
        "CV_accuracy": cv_accuracy,
        "R2_macro": r2_macro,
        "Q2_macro": q2_macro,
        "R2_per_class": {str(k): float(v) for k, v in r2_per_class.items()},
        "Q2_per_class": {str(k): float(v) for k, v in q2_per_class.items()},
        "R2_global_flat": r2_global_flat,
        "Q2_global_flat": q2_global_flat,
    }

    print("PLS-DA Double CV results:")
    print(f"  NMC: {nmc}")
    print(f"  AUROC: {auc}")
    print(f"  CV accuracy: {cv_accuracy:.4f}")
    print(f"  R2_macro: {r2_macro:.4f}")
    print(f"  Q2_macro: {q2_macro:.4f}")
    print("")

    state.plsda_response_column = response_column_names
    state.plsda_model = model
    state.plsda = model

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

    state.plsda_vip_scores.reset_index().rename(
        columns={"index": "cpdID", 0: "VIP_score"}
    ).to_csv(vip_path, index=False, sep=";")
    pd.DataFrame([state.plsda_stats]).to_csv(stats_path, index=False, sep=";")
    state.candidates.to_csv(candidates_path, index=False, sep=";")

    _add_artifact_if_available(state, vip_path, "table", "PLS-DA VIP scores")
    _add_artifact_if_available(state, stats_path, "table", "PLS-DA statistics")
    _add_artifact_if_available(
        state, candidates_path, "table", "Candidate features after PLS-DA"
    )

    if report is not None:
        text0 = (
            f"<b>PLS-DA</b> was performed with the {str(response_column_names)} "
            "column(s) as the response."
        )
        text1 = (
            "Double cross-validation was used for model validation "
            f"(outer splits: {outer_splits}, outer repeats: {outer_repeats}, "
            f"inner splits: {inner_splits}). LV selection metric: {select_metric}. "
            f"Final number of components: {n_comp}. AUROC: {auc:.4f}, NMC: {nmc}, "
            f"CV accuracy: {cv_accuracy:.4f}, R2_macro: {r2_macro:.4f}, "
            f"Q2_macro: {q2_macro:.4f}."
        )
        report.add_together([("text", text0), ("text", text1), "line"])

    return {
        "message": "PLS-DA was performed.",
        "warnings": warning_messages,
        "plsda_stats": state.plsda_stats,
        "vip_path": vip_path,
        "stats_path": stats_path,
        "candidates_path": candidates_path,
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
            help="Accepted for compatibility with old PySPRESSO. Old statistics_ttest did not apply the correction.",
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
    data = state.data.copy()
    metadata = state.metadata.copy()
    report = state.report
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
            "Running t-test cannot on scaled data; Please use unscaled data. Scaling affects variance and invalidates t-test assumptions."
        )

    if groups_column_name not in metadata.columns:
        raise ValueError(f"Column '{groups_column_name}' was not found in metadata.")

    group1_mask = metadata[metadata[groups_column_name] == group1].index.tolist()
    group2_mask = metadata[metadata[groups_column_name] == group2].index.tolist()

    group1_data = data.iloc[:, 1:].iloc[:, group1_mask]
    group2_data = data.iloc[:, 1:].iloc[:, group2_mask]

    g1_mean = group1_data.mean(axis=1)
    g2_mean = group2_data.mean(axis=1)
    if (g1_mean == 0).any() or (g2_mean == 0).any():
        group1_data += 1e-9
        group2_data += 1e-9

    if was_log_transformed:
        fold_change = log_base ** (g2_mean - g1_mean)
    else:
        fold_change = g2_mean / g1_mean

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

        p_values.append(p)
    p_values = pd.Series(p_values)

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
            "both groups normal": both_normal,
            "used test": tests_used,
            "group": [f"{group1} vs {group2}"] * len(data),
        }
    )

    p_values_table = p_values_table.sort_values(
        by="p-value", ascending=True
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

    if report is not None:
        text0 = f"<b>t-test</b> was performed for the groups: {group1} vs {group2}."
        text1 = "The results are shown in the table below."
        report.add_together(
            [("text", text0), ("text", text1), ("table", p_values_table), "line"]
        )
        report.add_text(f"The t-test results were saved to: {csv_name}")

    return {
        "message": "t-test / Mann-Whitney U was performed.",
        "warnings": warning_messages,
        "group1": group1,
        "group2": group2,
        "group1_n": len(group1_mask),
        "group2_n": len(group2_mask),
        "p_value_correction_method_parameter": p_value_correction_method,
        "table_path": csv_name,
    }


register_operation(
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

    color_by = str(color_by).strip()

    if color_by and state.metadata is not None and color_by in state.metadata.columns:
        metadata = state.metadata.copy()

        if "Sample File" not in metadata.columns:
            raise ValueError("Expected 'Sample File' column in metadata.")

        metadata = metadata[["Sample File", color_by]].copy()
        pca_df = pca_df.merge(metadata, on="Sample File", how="left")

        groups = pca_df[color_by].astype(str).fillna("Missing").unique()

        for group in groups:
            subset = pca_df[pca_df[color_by].astype(str).fillna("Missing") == group]

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

    plt.close(fig)

    if len(saved_paths) > 0:
        figure_base_path = os.path.splitext(saved_paths[0])[0]
    else:
        figure_base_path = base_path

    if state.report is not None:
        png_paths = [path for path in saved_paths if path.endswith(".png")]

        state.report.add_together(
            [
                ("text", "PCA score plot was created."),
                ("image", png_paths[0] if png_paths else saved_paths[0]),
                "line",
            ]
        )

    _add_artifact_if_available(
        state,
        saved_paths[0] if saved_paths else figure_base_path,
        "figure",
        "PCA score plot",
    )

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
    Port of old Workflow.visualizer_PCA_grouped() into GUI operation style.
    """

    print("TEST PRINT", flush=True)
    print("TEST PRINT", flush=True)
    print("TEST PRINT", flush=True)
    print("TEST PRINT", flush=True)
    print("TEST PRINT", flush=True)

    plt, sns = _load_plotting()

    import matplotlib as mpl
    from matplotlib.patches import Ellipse
    from scipy.stats import zscore
    from adjustText import adjust_text
    from itertools import cycle

    # GUI may pass "None" as a string, so normalize it to real None.
    if color_column in ["", "None", "none", "null", "NULL"]:
        color_column = None

    if marker_column in ["", "None", "none", "null", "NULL"]:
        marker_column = None

    if zoom_in_group in ["", "None", "none", "null", "NULL"]:
        zoom_in_group = None

    metadata = state.metadata.copy()
    original_metadata = state.metadata.copy()
    report = state.report

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

            print(type(v))
            print(v.dtype if hasattr(v, "dtype") else "no dtype")

            print(v)
            print(type(v[0, 0]))
            print(type(v[1, 0]))
            print(v[0, 0])
            print(v[1, 0])

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

    plt.close(fig)

    if len(saved_paths) > 0:
        figure_base_path = os.path.splitext(saved_paths[0])[0]
    else:
        figure_base_path = plt_name

    if column_name is not None and second_column_name is not None:
        text = (
            "Detailed PCA plot based on "
            + column_name
            + "(colors) and "
            + second_column_name
            + "(markers) was created and added into: "
            + figure_base_path
        )
    elif column_name is not None:
        text = (
            "Detailed PCA plot based on "
            + column_name
            + "(colors) was created and added into: "
            + figure_base_path
        )
    elif second_column_name is not None:
        text = (
            "Detailed PCA plot based on "
            + second_column_name
            + "(markers) was created and added into: "
            + figure_base_path
        )

    if report is not None:
        png_paths = [path for path in saved_paths if path.endswith(".png")]
        image_path = png_paths[0] if png_paths else saved_paths[0]

        report.add_together(
            [
                ("text", text),
                ("image", image_path),
            ]
        )

    return {
        "figure_base_path": figure_base_path,
        "saved_paths": saved_paths,
        "color_column": color_column,
        "marker_column": marker_column,
        "crossout_outliers": crossout_outliers,
        "zoom_in_group": zoom_in_group,
        "annotate_samples": annotate_samples,
    }
