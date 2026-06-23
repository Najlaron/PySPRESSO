from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum
import uuid
import pandas as pd
from pyspresso_app.config import db

# def save_workflow(workflow: Workflow, path: str) -> None:
#     with open(path, "wb") as f:
#         pickle.dump(workflow, f)

# def load_workflow(path: str) -> Workflow:
#     with open(path, "rb") as f:
#         return pickle.load(f)


@dataclass
class WorkflowState:
    workflow_id: str
    name: str
    pyspresso_version: str

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

    report: Any | None = None

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

    plsda: Any | None = None
    plsda_model: Any | None = None
    plsda_stats: dict[str, Any] | None = None
    plsda_metadata: pd.DataFrame | None = None
    plsda_response_column: str | list[str] | None = None
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "pyspresso_version": self.pyspresso_version,
            "data": (
                self.data.to_dict(orient="split") if self.data is not None else None
            ),
            "variable_metadata": (
                self.variable_metadata.to_dict(orient="split")
                if self.variable_metadata is not None
                else None
            ),
            "metadata": (
                self.metadata.to_dict(orient="split")
                if self.metadata is not None
                else None
            ),
            "batch_info": (
                self.batch_info.to_dict(orient="split")
                if self.batch_info is not None
                else None
            ),
            "batch": self.batch,
            "QC_samples": self.QC_samples,
            "blank_samples": self.blank_samples,
            "standard_samples": self.standard_samples,
            "dilution_series_samples": self.dilution_series_samples,
            "dil_concentrations": self.dil_concentrations,
            "report": self.report,
            "was_log_transformed": self.was_log_transformed,
            "log_base": self.log_base,
            "was_centered": self.was_centered,
            "was_scaled": self.was_scaled,
            "was_normalized": self.was_normalized,
            "saves_count": self.saves_count,
            "pca_count": self.pca_count,
            "fold_change_count": self.fold_change_count,
            "pca": self.pca,
            "pca_data": self.pca_data,
            "pca_df": (
                self.pca_df.to_dict(orient="split") if self.pca_df is not None else None
            ),
            "pca_per_var": self.pca_per_var,
            "pca_loadings": (
                self.pca_loadings.to_dict(orient="split")
                if self.pca_loadings is not None
                else None
            ),
            "pca_loadings_candidates": self.pca_loadings_candidates,
            "pca_loadings_candidates_len": self.pca_loadings_candidates_len,
            "fold_change": (
                self.fold_change.to_dict(orient="split")
                if self.fold_change is not None
                else None
            ),
            "plsda": self.plsda,
            "plsda_model": self.plsda_model,
            "plsda_stats": self.plsda_stats,
            "plsda_metadata": (
                self.plsda_metadata.to_dict(orient="split")
                if self.plsda_metadata is not None
                else None
            ),
            "plsda_response_column": self.plsda_response_column,
            "plsda_vip_scores": (
                self.plsda_vip_scores.to_dict()
                if self.plsda_vip_scores is not None
                else None
            ),
            "plsda_vip_candidates": self.plsda_vip_candidates,
            "plsda_vip_candidates_len": self.plsda_vip_candidates_len,
            "candidates": self.candidates.to_dict(orient="split"),
            "execution_log": self.execution_log,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        """Deserialize WorkflowState from dictionary (e.g., from database JSON)."""

        def _restore_dataframe(df_dict):
            if df_dict is None:
                return None
            # orient="split" format returns dict with 'index', 'columns', 'data'
            return pd.DataFrame(
                df_dict.get("data"),
                index=df_dict.get("index"),
                columns=df_dict.get("columns"),
            )

        def _restore_series(series_dict):
            if series_dict is None:
                return None
            return pd.Series(series_dict)

        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            pyspresso_version=data["pyspresso_version"],
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
            report=data.get("report"),
            was_log_transformed=data.get("was_log_transformed", False),
            log_base=data.get("log_base"),
            was_centered=data.get("was_centered", False),
            was_scaled=data.get("was_scaled"),
            was_normalized=data.get("was_normalized"),
            saves_count=data.get("saves_count", 0),
            pca_count=data.get("pca_count", 0),
            fold_change_count=data.get("fold_change_count", 0),
            pca=data.get("pca"),
            pca_data=data.get("pca_data"),
            pca_df=_restore_dataframe(data.get("pca_df")),
            pca_per_var=data.get("pca_per_var"),
            pca_loadings=_restore_dataframe(data.get("pca_loadings")),
            pca_loadings_candidates=data.get("pca_loadings_candidates"),
            pca_loadings_candidates_len=data.get("pca_loadings_candidates_len"),
            fold_change=_restore_dataframe(data.get("fold_change")),
            plsda=data.get("plsda"),
            plsda_model=data.get("plsda_model"),
            plsda_stats=data.get("plsda_stats"),
            plsda_metadata=_restore_dataframe(data.get("plsda_metadata")),
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

    files: dict[str, str] = field(default_factory=dict)
    steps: list[WorkflowStep] = field(default_factory=list)

    valid: bool = True
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "pyspresso_version": self.pyspresso_version,
            "files": self.files,
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
            files=data.get("files", {}),
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
