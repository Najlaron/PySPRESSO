from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import math
import re
from pyspresso_app.config import db


def _safe_convert_value(x: Any) -> Any:
    """Convert a single value to a JSON-serializable equivalent.

    - NaN/Infinity -> None
    - pd.Timestamp / objects with isoformat -> isoformat string
    - otherwise return value as-is
    """
    if x is None or isinstance(x, (str, bool, int)):
        return x

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass

    if isinstance(x, dict):
        return {str(key): _safe_convert_value(value) for key, value in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_safe_convert_value(value) for value in x]

    if hasattr(x, "item"):
        try:
            return _safe_convert_value(x.item())
        except Exception:
            pass

    # Match json_safe's safe fallback for uncommon object values.
    return f"<{type(x).__name__}>"


def _convert_datetime_col(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.isoformat() if pd.notnull(x) else None)


def _convert_generic_col(series: pd.Series) -> pd.Series:
    return series.apply(_safe_convert_value)


def _df_to_serializable(df: pd.DataFrame | None) -> Any:
    """Convert DataFrame to JSON-serializable dict, handling NaN, Infinity, and null values.

    Numeric columns are copied to an object array in vectorized code, preserving
    Python's full float representation. Only object and datetime columns need a
    Python-level conversion. The former all-column per-cell ``apply`` path became
    a noticeable part of every workflow operation for larger tables.
    """
    if df is None:
        return None

    values = df.to_numpy(dtype=object, na_value=None)

    for position, dtype in enumerate(df.dtypes):
        if pd.api.types.is_datetime64_any_dtype(dtype):
            values[:, position] = [
                item.isoformat() if item is not None else None
                for item in values[:, position]
            ]
        elif pd.api.types.is_float_dtype(dtype):
            numeric_values = df.iloc[:, position].to_numpy(
                dtype=float,
                na_value=np.nan,
            )
            values[~np.isfinite(numeric_values), position] = None
        elif (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            values[:, position] = [
                _safe_convert_value(item) for item in values[:, position]
            ]

    return {
        "index": [_safe_convert_value(item) for item in df.index.tolist()],
        "columns": [_safe_convert_value(item) for item in df.columns.tolist()],
        "data": values.tolist(),
    }


def _series_to_serializable(series: pd.Series | None) -> Any:
    """Convert Series to JSON-serializable dict, handling NaN, Infinity, and null values."""
    if series is None:
        return None
    s2 = series.copy()
    # Delegate conversion to the shared helper for consistency
    s2 = s2.apply(_safe_convert_value)
    return s2.to_dict()


def _contains_legacy_truncation(value: Any) -> bool:
    """Detect the former preview marker without walking every table cell.

    The old serializer always appended its marker at a container boundary. For
    lists it is therefore sufficient to inspect the first and final values. A
    recursive scan of every value made every workflow load proportional to the
    complete dataset before an operation had even started.
    """
    if isinstance(value, dict):
        if "_truncated" in value:
            return True
        return any(_contains_legacy_truncation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        boundary_values = (value[0], value[-1]) if len(value) > 1 else (value[0],)
        return any(_contains_legacy_truncation(item) for item in boundary_values)
    if isinstance(value, str):
        return "additional items omitted>" in value or "additional keys omitted" in value
    return False


_ISO_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def _looks_like_serialized_datetime(value: Any) -> bool:
    """Return whether a value matches the ISO form used by ``_df_to_serializable``."""
    return isinstance(value, str) and bool(
        _ISO_DATETIME_PATTERN.fullmatch(value.strip())
    )


def _should_restore_datetime(values: pd.Series | pd.Index) -> bool:
    """Cheaply identify serialized datetimes before invoking pandas' parser."""
    non_null = pd.Series(values).dropna()
    if non_null.empty:
        return False
    sample = non_null.iloc[: min(len(non_null), 20)]
    return bool(sample.map(_looks_like_serialized_datetime).all())


@dataclass
class WorkflowState:
    workflow_id: str
    name: str
    pyspresso_version: str
    main_folder: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    data: pd.DataFrame | None = None
    variable_metadata: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    batch_info: pd.DataFrame | None = None

    batch: list[str] | None = None
    QC_samples: list[str] | None = None
    blank_samples: list[str] | bool | None = None
    standard_samples: list[str] | bool | None = None
    dilution_series_samples: list[str] | bool | None = None
    dil_concentrations: list[float] | None = None

    # Deprecated runtime-only PDF reporter object. Do not serialize this.
    report: Any | None = None

    # Live HTML reporting. These fields are JSON-safe and can be stored in state.
    report_file_name: str | None = None
    report_manifest_path: str | None = None
    report_html_path: str | None = None
    report_html_url: str | None = None
    report_entries: list[dict[str, Any]] = field(default_factory=list)

    was_log_transformed: bool = False
    log_base: float | int | None = None
    was_centered: bool = False
    was_scaled: str | bool | None = False
    was_normalized: str | bool | None = False

    saves_count: int = 0
    pca_count: int = 0
    fold_change_count: int = 0

    pca: Any | None = None
    pca_data: Any | None = None
    pca_df: pd.DataFrame | None = None
    pca_per_var: Any | None = None
    pca_loadings: pd.DataFrame | None = None
    pca_loadings_candidates: Any | None = None
    pca_loadings_candidates_len: int | None = None

    fold_change: pd.DataFrame | None = None

    # PLS-DA runtime / result state
    # Runtime sklearn objects are never serialized to JSON.
    plsda: Any | None = None
    plsda_model: Any | None = None

    # JSON-safe PLS-DA outputs / metadata
    plsda_model_path: str | None = None
    plsda_scores_path: str | None = None
    plsda_vip_scores_path: str | None = None
    plsda_cv_predictions_path: str | None = None
    plsda_class_names: list[str] | None = None
    plsda_feature_ids: list[str] | None = None
    plsda_score_columns: list[str] | None = None

    plsda_stats: dict[str, Any] | None = None
    plsda_metadata: pd.DataFrame | None = None
    plsda_scores: pd.DataFrame | None = None
    plsda_response_column: Any | None = None
    plsda_vip_scores: pd.Series | None = None
    plsda_vip_candidates: Any | None = None
    plsda_vip_candidates_len: int | None = None

    candidates: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["feature", "method", "specification", "score", "hits"]
        )
    )

    execution_log: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    state_integrity_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "pyspresso_version": self.pyspresso_version,
            "main_folder": self.main_folder,
            "files": self.files,
            "data": (_df_to_serializable(self.data) if self.data is not None else None),
            "variable_metadata": (
                _df_to_serializable(self.variable_metadata)
                if self.variable_metadata is not None
                else None
            ),
            "metadata": (
                _df_to_serializable(self.metadata)
                if self.metadata is not None
                else None
            ),
            "batch_info": (
                _df_to_serializable(self.batch_info)
                if self.batch_info is not None
                else None
            ),
            "batch": self.batch,
            "QC_samples": self.QC_samples,
            "blank_samples": self.blank_samples,
            "standard_samples": self.standard_samples,
            "dilution_series_samples": self.dilution_series_samples,
            "dil_concentrations": self.dil_concentrations,
            "report": None,
            "report_file_name": self.report_file_name,
            "report_manifest_path": self.report_manifest_path,
            "report_html_path": self.report_html_path,
            "report_html_url": self.report_html_url,
            "report_entries": self.report_entries,
            "was_log_transformed": self.was_log_transformed,
            "log_base": self.log_base,
            "was_centered": self.was_centered,
            "was_scaled": self.was_scaled,
            "was_normalized": self.was_normalized,
            "saves_count": self.saves_count,
            "pca_count": self.pca_count,
            "fold_change_count": self.fold_change_count,
            "pca": None,
            "pca_data": (
                self.pca_data.tolist()
                if hasattr(self.pca_data, "tolist")
                else self.pca_data
            ),
            "pca_df": (
                _df_to_serializable(self.pca_df) if self.pca_df is not None else None
            ),
            "pca_per_var": self.pca_per_var,
            "pca_loadings": (
                _df_to_serializable(self.pca_loadings)
                if self.pca_loadings is not None
                else None
            ),
            "pca_loadings_candidates": self.pca_loadings_candidates,
            "pca_loadings_candidates_len": self.pca_loadings_candidates_len,
            "fold_change": (
                _df_to_serializable(self.fold_change)
                if self.fold_change is not None
                else None
            ),
            # Runtime-only PLS-DA objects are not JSON-serialized.
            "plsda": None,
            "plsda_model": None,

            # JSON-safe PLS-DA file paths / metadata
            "plsda_model_path": self.plsda_model_path,
            "plsda_scores_path": self.plsda_scores_path,
            "plsda_vip_scores_path": self.plsda_vip_scores_path,
            "plsda_cv_predictions_path": self.plsda_cv_predictions_path,
            "plsda_class_names": self.plsda_class_names,
            "plsda_feature_ids": self.plsda_feature_ids,
            "plsda_score_columns": self.plsda_score_columns,

            # JSON-safe PLS-DA results
            "plsda_stats": self.plsda_stats,
            "plsda_metadata": (
                _df_to_serializable(self.plsda_metadata)
                if self.plsda_metadata is not None
                else None
            ),
            "plsda_scores": (
                _df_to_serializable(self.plsda_scores)
                if self.plsda_scores is not None
                else None
            ),
            "plsda_response_column": self.plsda_response_column,
            "plsda_vip_scores": (
                _series_to_serializable(self.plsda_vip_scores)
                if self.plsda_vip_scores is not None
                else None
            ),
            "plsda_vip_candidates": self.plsda_vip_candidates,
            "plsda_vip_candidates_len": self.plsda_vip_candidates_len,

            "candidates": _df_to_serializable(self.candidates),
            "execution_log": self.execution_log,
            "artifacts": self.artifacts,
            "state_integrity_error": self.state_integrity_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        """Deserialize WorkflowState from dictionary (e.g., from database JSON)."""

        state_integrity_error = data.get("state_integrity_error")
        persisted_tables = (
            data.get("data"),
            data.get("metadata"),
            data.get("variable_metadata"),
            data.get("batch_info"),
        )
        if state_integrity_error is None and any(
            _contains_legacy_truncation(table) for table in persisted_tables
        ):
            state_integrity_error = (
                "This workflow was saved with the former 80-item truncation bug. "
                "Its omitted samples/features cannot be recovered from the database; "
                "create a new workflow from the original input files."
            )

        def _restore_dataframe(df_dict):
            if df_dict is None:
                return None

            if isinstance(df_dict, dict) and {"data", "columns"}.issubset(df_dict.keys()):
                rows = df_dict.get("data") or []
                columns = list(df_dict.get("columns") or [])
                index = df_dict.get("index")

                if columns and isinstance(rows, list) and any(
                    not isinstance(row, list) or len(row) != len(columns)
                    for row in rows
                ):
                    n_cols = len(columns)
                    repaired_rows = []

                    for row in rows:
                        if isinstance(row, list):
                            if len(row) > n_cols:
                                repaired_rows.append(row[:n_cols])
                            elif len(row) < n_cols:
                                repaired_rows.append(row + [None] * (n_cols - len(row)))
                            else:
                                repaired_rows.append(row)
                        else:
                            repaired_rows.append([row] + [None] * (n_cols - 1))

                    rows = repaired_rows

                try:
                    df = pd.DataFrame(rows, index=index, columns=columns or None)
                except ValueError:
                    # Last-resort fallback: load without explicit column labels, then
                    # keep only the columns that were actually stored in the split
                    # metadata. Do NOT create extra_* columns, because those can be
                    # mistaken for real LC-MS sample columns by downstream methods.
                    df = pd.DataFrame(rows, index=index)
                    if columns:
                        if len(columns) < df.shape[1]:
                            df = df.iloc[:, : len(columns)].copy()
                        df.columns = columns[: df.shape[1]]

                generated_extra_cols = [
                    col
                    for col in df.columns
                    if str(col).lower().startswith("extra_")
                    or (
                        "additional" in str(col).lower()
                        and (str(col).startswith("<") or "item" in str(col).lower())
                    )
                ]

                if generated_extra_cols:
                    df = df.drop(columns=generated_extra_cols)

            else:
                # Compatibility fallback for potential non-split serialized states.
                df = pd.DataFrame(df_dict)

            try:
                idx = pd.Index(df.index)
                if _should_restore_datetime(idx):
                    parsed_idx = pd.to_datetime(idx, errors="coerce", format="mixed")
                    if parsed_idx.notna().all():
                        df.index = parsed_idx
            except Exception:
                pass

            for col in df.columns:
                ser = df[col]
                non_null = ser.dropna()

                if non_null.empty:
                    continue

                if _should_restore_datetime(non_null):
                    parsed = pd.to_datetime(ser, errors="coerce", format="mixed")
                    if parsed[ser.notna()].notna().all():
                        df[col] = parsed

            return df

        def _restore_series(series_dict):
            if series_dict is None:
                return None
            return pd.Series(series_dict)

        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            pyspresso_version=data["pyspresso_version"],
            main_folder=data.get("main_folder"),
            files=data.get("files", {}),
            data=_restore_dataframe(data.get("data")),
            variable_metadata=_restore_dataframe(data.get("variable_metadata")),
            metadata=_restore_dataframe(data.get("metadata")),
            batch_info=_restore_dataframe(data.get("batch_info")),
            batch=data.get("batch"),
            QC_samples=data.get("QC_samples"),
            blank_samples=data.get("blank_samples"),
            standard_samples=data.get("standard_samples"),
            dilution_series_samples=data.get("dilution_series_samples"),
            dil_concentrations=data.get("dil_concentrations"),
            report=None,
            report_file_name=data.get("report_file_name"),
            report_manifest_path=data.get("report_manifest_path"),
            report_html_path=data.get("report_html_path"),
            report_html_url=data.get("report_html_url"),
            report_entries=data.get("report_entries", []),
            was_log_transformed=data.get("was_log_transformed", False),
            log_base=data.get("log_base"),
            was_centered=data.get("was_centered", False),
            was_scaled=data.get("was_scaled"),
            was_normalized=data.get("was_normalized"),
            saves_count=data.get("saves_count", 0),
            pca_count=data.get("pca_count", 0),
            fold_change_count=data.get("fold_change_count", 0),
            pca=None,
            pca_data=data.get("pca_data"),
            pca_df=_restore_dataframe(data.get("pca_df")),
            pca_per_var=data.get("pca_per_var"),
            pca_loadings=_restore_dataframe(data.get("pca_loadings")),
            pca_loadings_candidates=data.get("pca_loadings_candidates"),
            pca_loadings_candidates_len=data.get("pca_loadings_candidates_len"),
            fold_change=_restore_dataframe(data.get("fold_change")),
            # Runtime-only PLS-DA objects are intentionally not restored from JSON.
            # The fitted model is saved as a .pkl file and only the path is restored.
            plsda=None,
            plsda_model=None,

            plsda_model_path=data.get("plsda_model_path"),
            plsda_scores_path=data.get("plsda_scores_path"),
            plsda_vip_scores_path=data.get("plsda_vip_scores_path"),
            plsda_cv_predictions_path=data.get("plsda_cv_predictions_path"),
            plsda_class_names=data.get("plsda_class_names"),
            plsda_feature_ids=data.get("plsda_feature_ids"),
            plsda_score_columns=data.get("plsda_score_columns"),

            plsda_stats=data.get("plsda_stats"),
            plsda_metadata=_restore_dataframe(data.get("plsda_metadata")),
            plsda_scores=_restore_dataframe(data.get("plsda_scores")),
            plsda_response_column=data.get("plsda_response_column"),
            plsda_vip_scores=_restore_series(data.get("plsda_vip_scores")),
            plsda_vip_candidates=data.get("plsda_vip_candidates"),
            plsda_vip_candidates_len=data.get("plsda_vip_candidates_len"),

            candidates=(
                _restore_dataframe(data.get("candidates"))
                if data.get("candidates")
                else pd.DataFrame(
                    columns=["feature", "method", "specification", "score", "hits"]
                )
            ),
            execution_log=data.get("execution_log", []),
            artifacts=data.get("artifacts", []),
            state_integrity_error=state_integrity_error,
        )


class StepStatus(str, Enum):
    NOT_RUN = "not_run"
    READY = "ready"
    NEEDS_PARAMETERS = "needs_parameters"
    BLOCKED = "blocked"
    RUNNING = "running"
    DONE = "done"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    step_id: str
    operation_id: str
    params: dict[str, Any] = field(default_factory=dict)

    enabled: bool = True
    valid: bool = True
    status: StepStatus = StepStatus.NOT_RUN

    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation_id": self.operation_id,
            "params": self.params,
            "enabled": self.enabled,
            "valid": self.valid,
            "status": self.status.value,
            "messages": self.messages,
            "warnings": self.warnings,
            "output_summary": self.output_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        return cls(
            step_id=data["step_id"],
            operation_id=data["operation_id"],
            params=data.get("params", {}),
            enabled=data.get("enabled", True),
            valid=data.get("valid", True),
            status=StepStatus(data.get("status", "not_run")),
            messages=data.get("messages", []),
            warnings=data.get("warnings", []),
            output_summary=data.get("output_summary", {}),
        )


@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    pyspresso_version: str

    steps: list[WorkflowStep] = field(default_factory=list)

    valid: bool = True
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "pyspresso_version": self.pyspresso_version,
            "steps": [step.to_dict() for step in self.steps],
            "valid": self.valid,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowDefinition":
        """Deserialize WorkflowDefinition from dictionary (e.g., from database JSON)."""
        steps = [
            WorkflowStep.from_dict(step_data) for step_data in data.get("steps", [])
        ]
        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            pyspresso_version=data["pyspresso_version"],
            steps=steps,
            valid=data.get("valid", True),
            messages=data.get("messages", []),
        )


@dataclass
class Workflow:
    workflow_id: str
    name: str = "PySPRESSO_Workflow"
    pyspresso_version: str = "0.0.5"

    definition: WorkflowDefinition = field(init=False)
    state: WorkflowState = field(init=False)

    def __post_init__(self):
        self.definition = WorkflowDefinition(
            workflow_id=self.workflow_id,
            name=self.name,
            pyspresso_version=self.pyspresso_version,
        )
        self.state = WorkflowState(
            workflow_id=self.workflow_id,
            name=self.name,
            pyspresso_version=self.pyspresso_version,
        )

    @property
    def steps(self) -> list[WorkflowStep]:
        return self.definition.steps


class WorkflowORM(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_name = db.Column(db.String(80), unique=False, nullable=False)
    pyspresso_version = db.Column(db.String(80), nullable=False)

    definition = db.Column(db.JSON, nullable=False)
    state = db.Column(db.JSON, nullable=False)

    folder_name = db.Column(db.String(255), nullable=True)
    report_file_name = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def to_dict(self):
        return {
            "id": self.id,
            "workflow_name": self.workflow_name,
            "pyspresso_version": self.pyspresso_version,
            "definition": self.definition,
            "state": self.state,
            "folder_name": self.folder_name,
            "report_file_name": self.report_file_name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
