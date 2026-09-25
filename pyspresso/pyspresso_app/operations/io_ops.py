# pyspresso_app/operations/io_ops.py

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone

import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState
from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)

# Helper functions
def _safe_checkpoint_name(value) -> str:
    """
    Convert a user-provided checkpoint name into a safe directory name.
    """

    name = str(value or "").strip()

    if not name:
        raise ValueError(
            "Checkpoint name cannot be empty."
        )

    # Windows-invalid filename characters.
    name = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    name = name.strip("._")

    if not name:
        raise ValueError(
            "Checkpoint name does not contain any valid characters."
        )

    return name


def _get_data_checkpoints_folder(
    state: WorkflowState,
) -> str:
    """
    Return the workflow's data_checkpoints directory.

    The initializer should already create this folder, but the IO
    operation also creates it defensively if necessary.
    """

    main_folder = getattr(
        state,
        "main_folder",
        None,
    )

    if not main_folder:
        workflow_id = getattr(
            state,
            "workflow_id",
            "workflow",
        )

        main_folder = os.path.join(
            "outputs",
            str(workflow_id),
        )

        state.main_folder = main_folder

    checkpoints_folder = os.path.join(
        main_folder,
        "data_checkpoints",
    )

    os.makedirs(
        checkpoints_folder,
        exist_ok=True,
    )

    return checkpoints_folder


def _save_dataframe(
    dataframe,
    path: str,
) -> bool:
    """
    Save a DataFrame as semicolon-separated CSV.

    Returns True if a table was saved and False if the value was None.
    Any stale file from an older version of the checkpoint is removed.
    """

    if dataframe is None:
        if os.path.exists(path):
            os.remove(path)

        return False

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame for {path}, "
            f"got {type(dataframe).__name__}."
        )

    dataframe.reset_index(
        drop=True
    ).to_csv(
        path,
        sep=";",
        index=False,
    )

    return True


def _json_safe(value):
    """
    Convert simple workflow-state values to JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, bool, int, float),
    ):
        return value

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    # numpy scalar and similar objects
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    # Timestamp / datetime-like objects
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return str(value)


def _clear_derived_statistics(
    state: WorkflowState,
):
    """
    Clear statistical results that belong to the data state that was
    active before a checkpoint was loaded.

    Analysis counters are deliberately NOT reset. They should continue
    increasing so later analyses remain uniquely numbered.
    """

    fields_to_clear = [
        # PCA
        "pca",
        "pca_data",
        "pca_df",
        "pca_per_var",
        "pca_loadings",
        "pca_loadings_candidates",
        "pca_loadings_candidates_len",

        # PLS-DA
        "plsda",
        "plsda_model",
        "plsda_stats",
        "plsda_metadata",
        "plsda_response_column",
        "plsda_scores",
        "plsda_class_names",
        "plsda_feature_ids",
        "plsda_score_columns",
        "plsda_vip_scores",
        "plsda_vip_candidates",
        "plsda_vip_candidates_len",
        "plsda_model_path",
        "plsda_scores_path",
        "plsda_vip_scores_path",
        "plsda_cv_predictions_path",

        # Other statistics tied to the previous data state
        "fold_change",
    ]

    for field_name in fields_to_clear:
        if hasattr(state, field_name):
            setattr(
                state,
                field_name,
                None,
            )


# Save checkpoint
@register_operation(
    id="save_data_checkpoint",
    label="Save Data Checkpoint",
    description=(
        "Save the current data state under a checkpoint name so it can "
        "later be restored. Useful for branching an analysis before "
        "filtering, correction, transformation, or other processing."
    ),
    citation="",
    category_tags=[
        OperationTag.IO,
    ],
    parameter_schema=[
        ParameterDef(
            name="checkpoint_name",
            type="str",
            required=True,
            default="",
            label="Checkpoint name",
            help=(
                "Name used to identify this checkpoint later. "
                "For example: before_filter, after_qc_correction."
            ),
            example="before_filter",
        ),
    ],
    requires=[
        "data",
        "metadata",
        "variable_metadata",
    ],
    produces=[],
)
def save_data_checkpoint(
    state: WorkflowState,
    checkpoint_name: str,
):
    """
    Save the current data state.

    Restorable:
        - data
        - metadata
        - variable_metadata
        - batch_info
        - batch labels
        - sample-type lists
        - dilution concentrations
        - transformation/scaling state

    Candidates are saved as a snapshot for provenance, but are NOT
    restored when the checkpoint is loaded.
    """

    checkpoint_name = _safe_checkpoint_name(
        checkpoint_name
    )

    checkpoints_folder = (
        _get_data_checkpoints_folder(state)
    )

    checkpoint_dir = os.path.join(
        checkpoints_folder,
        checkpoint_name,
    )

    # Re-running the same workflow step should update the checkpoint
    # instead of mixing new files with stale files from an older save.
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    # Dataset files
    paths = {
        "data": os.path.join(
            checkpoint_dir,
            "data.csv",
        ),
        "metadata": os.path.join(
            checkpoint_dir,
            "metadata.csv",
        ),
        "variable_metadata": os.path.join(
            checkpoint_dir,
            "variable_metadata.csv",
        ),
        "batch_info": os.path.join(
            checkpoint_dir,
            "batch_info.csv",
        ),
        "candidates": os.path.join(
            checkpoint_dir,
            "candidates.csv",
        ),
    }

    _save_dataframe(
        state.data,
        paths["data"],
    )

    _save_dataframe(
        state.metadata,
        paths["metadata"],
    )

    _save_dataframe(
        state.variable_metadata,
        paths["variable_metadata"],
    )

    batch_info_saved = _save_dataframe(
        getattr(
            state,
            "batch_info",
            None,
        ),
        paths["batch_info"],
    )

    # Candidates are deliberately saved, but will not be restored.
    candidates_saved = _save_dataframe(
        getattr(
            state,
            "candidates",
            None,
        ),
        paths["candidates"],
    )


    # Non-table state
    auxiliary_state = {
        "batch": getattr(
            state,
            "batch",
            None,
        ),
        "QC_samples": getattr(
            state,
            "QC_samples",
            None,
        ),
        "blank_samples": getattr(
            state,
            "blank_samples",
            None,
        ),
        "standard_samples": getattr(
            state,
            "standard_samples",
            None,
        ),
        "dilution_series_samples": getattr(
            state,
            "dilution_series_samples",
            None,
        ),
        "dil_concentrations": getattr(
            state,
            "dil_concentrations",
            None,
        ),
        "was_log_transformed": getattr(
            state,
            "was_log_transformed",
            False,
        ),
        "log_base": getattr(
            state,
            "log_base",
            None,
        ),
        "was_centered": getattr(
            state,
            "was_centered",
            False,
        ),
        "was_scaled": getattr(
            state,
            "was_scaled",
            False,
        ),
        "was_normalized": getattr(
            state,
            "was_normalized",
            False,
        ),
    }

    manifest = {
        "checkpoint_name": checkpoint_name,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "n_features": int(
            state.data.shape[0]
        ),
        "n_samples": int(
            state.data.shape[1] - 1
        ),
        "batch_info_saved": batch_info_saved,
        "candidates_saved": candidates_saved,
        "auxiliary_state": _json_safe(
            auxiliary_state
        ),
    }

    manifest_path = os.path.join(
        checkpoint_dir,
        "checkpoint.json",
    )

    # Write manifest LAST.
    # Therefore a checkpoint without a manifest is considered incomplete.
    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Data checkpoint '{checkpoint_name}' was saved.\n"
            f"Features: {state.data.shape[0]}\n"
            f"Samples: {state.data.shape[1] - 1}\n"
            f"Candidates snapshot saved: {candidates_saved}"
        ),
        title="Data checkpoint saved",
        preformatted=True,
    )

    return {
        "checkpoint_name": checkpoint_name,
        "checkpoint_directory": checkpoint_dir,
        "n_features": int(
            state.data.shape[0]
        ),
        "n_samples": int(
            state.data.shape[1] - 1
        ),
        "candidates_saved": candidates_saved,
    }


# Load checkpoint
@register_operation(
    id="load_data_checkpoint",
    label="Load Data Checkpoint",
    description=(
        "Restore data and associated preprocessing state from a previously "
        "saved checkpoint. Candidate features accumulated during the current "
        "workflow are preserved and are not replaced by the checkpoint."
    ),
    citation="",
    category_tags=[
        OperationTag.IO,
    ],
    parameter_schema=[
        ParameterDef(
            name="checkpoint_name",
            type="str",
            required=True,
            default="",
            label="Checkpoint name",
            help=(
                "Name of a checkpoint previously created using "
                "'Save Data Checkpoint'."
            ),
            example="before_filter",
        ),
    ],
    requires=[],
    produces=[
        "data",
        "metadata",
        "variable_metadata",
        "batch_info",
        "batch",
        "QC_samples",
        "blank_samples",
        "standard_samples",
        "dilution_series_samples",
        "dil_concentrations",
        "was_log_transformed",
        "log_base",
        "was_centered",
        "was_scaled",
        "was_normalized",
    ],
)
def load_data_checkpoint(
    state: WorkflowState,
    checkpoint_name: str,
):
    """
    Restore a previously saved data checkpoint.

    Candidates are deliberately NOT restored.

    PCA, PLS-DA and fold-change results are cleared because they may
    have been calculated on a different data state.
    """

    checkpoint_name = _safe_checkpoint_name(
        checkpoint_name
    )

    checkpoints_folder = (
        _get_data_checkpoints_folder(state)
    )

    checkpoint_dir = os.path.join(
        checkpoints_folder,
        checkpoint_name,
    )

    manifest_path = os.path.join(
        checkpoint_dir,
        "checkpoint.json",
    )

    if not os.path.isdir(checkpoint_dir):
        available = sorted(
            name
            for name in os.listdir(
                checkpoints_folder
            )
            if os.path.isdir(
                os.path.join(
                    checkpoints_folder,
                    name,
                )
            )
        )

        available_text = (
            ", ".join(available)
            if available
            else "none"
        )

        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' was not found. "
            f"Available checkpoints: {available_text}."
        )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' is incomplete: "
            "checkpoint.json is missing."
        )

    paths = {
        "data": os.path.join(
            checkpoint_dir,
            "data.csv",
        ),
        "metadata": os.path.join(
            checkpoint_dir,
            "metadata.csv",
        ),
        "variable_metadata": os.path.join(
            checkpoint_dir,
            "variable_metadata.csv",
        ),
        "batch_info": os.path.join(
            checkpoint_dir,
            "batch_info.csv",
        ),
    }

    required_files = [
        paths["data"],
        paths["metadata"],
        paths["variable_metadata"],
    ]

    missing_files = [
        os.path.basename(path)
        for path in required_files
        if not os.path.exists(path)
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' is incomplete. "
            f"Missing file(s): {', '.join(missing_files)}."
        )

    # Read manifest before changing the current state.
    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)


    # Load tables first into temporary variables.
    restored_data = pd.read_csv(
        paths["data"],
        sep=";",
    )

    restored_metadata = pd.read_csv(
        paths["metadata"],
        sep=";",
    )

    restored_variable_metadata = pd.read_csv(
        paths["variable_metadata"],
        sep=";",
    )

    if os.path.exists(
        paths["batch_info"]
    ):
        restored_batch_info = pd.read_csv(
            paths["batch_info"],
            sep=";",
        )
    else:
        restored_batch_info = None

    # Basic integrity checks.
    if restored_data.shape[1] < 2:
        raise ValueError(
            f"Checkpoint '{checkpoint_name}' contains an invalid "
            "data matrix with fewer than two columns."
        )

    expected_features = manifest.get(
        "n_features"
    )

    expected_samples = manifest.get(
        "n_samples"
    )

    actual_features = int(
        restored_data.shape[0]
    )

    actual_samples = int(
        restored_data.shape[1] - 1
    )

    if (
        expected_features is not None
        and int(expected_features)
        != actual_features
    ):
        raise ValueError(
            f"Checkpoint '{checkpoint_name}' failed integrity check: "
            f"expected {expected_features} features, "
            f"found {actual_features}."
        )

    if (
        expected_samples is not None
        and int(expected_samples)
        != actual_samples
    ):
        raise ValueError(
            f"Checkpoint '{checkpoint_name}' failed integrity check: "
            f"expected {expected_samples} samples, "
            f"found {actual_samples}."
        )

    # Restore active datasets.
    state.data = restored_data
    state.metadata = restored_metadata
    state.variable_metadata = (
        restored_variable_metadata
    )
    state.batch_info = restored_batch_info


    # Restore preprocessing / sample state.
    auxiliary_state = manifest.get(
        "auxiliary_state",
        {},
    )

    fields_to_restore = [
        "batch",
        "QC_samples",
        "blank_samples",
        "standard_samples",
        "dilution_series_samples",
        "dil_concentrations",
        "was_log_transformed",
        "log_base",
        "was_centered",
        "was_scaled",
        "was_normalized",
    ]

    for field_name in fields_to_restore:
        if field_name in auxiliary_state:
            setattr(
                state,
                field_name,
                auxiliary_state[field_name],
            )

    # ------------------------------------------------------------------
    # Candidates are intentionally preserved.
    #
    # The candidates.csv file inside the checkpoint is only a historical
    # snapshot showing which candidates existed when the checkpoint was
    # created.
    # ------------------------------------------------------------------

    # Do NOT assign state.candidates here.

    _clear_derived_statistics(state)

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Data checkpoint '{checkpoint_name}' was restored.\n"
            f"Features: {state.data.shape[0]}\n"
            f"Samples: {state.data.shape[1] - 1}\n"
            "Candidates accumulated in the workflow were preserved.\n"
            "PCA, PLS-DA and fold-change results from the previous "
            "active data state were cleared."
        ),
        title="Data checkpoint loaded",
        preformatted=True,
    )

    return {
        "checkpoint_name": checkpoint_name,
        "checkpoint_directory": checkpoint_dir,
        "n_features": int(
            state.data.shape[0]
        ),
        "n_samples": int(
            state.data.shape[1] - 1
        ),
        "candidates_restored": False,
        "derived_statistics_cleared": True,
    }