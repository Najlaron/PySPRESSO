from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import pandas as pd


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

@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    pyspresso_version: str

    files: dict[str, str] = field(default_factory=dict)
    steps: list[WorkflowStep] = field(default_factory=list)

    valid: bool = True
    messages: list[str] = field(default_factory=list)

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