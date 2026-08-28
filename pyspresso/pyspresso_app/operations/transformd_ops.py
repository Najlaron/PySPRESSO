import numpy as np
import pandas as pd

from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState


@register_operation(
    id="transformer_log",
    label="Log Transform",
    description="Apply or invert log transformation on sample intensity columns.",
    citation="",
    category_tags=[OperationTag.TRANSFORMATION],
    parameter_schema=[
        ParameterDef("base", "float", False, 2, "Log base"),
        ParameterDef("invert", "bool", False, False, "Invert transform"),
    ],
    requires=["data"],
    produces=["data", "was_log_transformed", "log_base"],
)
def transformer_log(
    state: WorkflowState,
    base: float = 2,
    invert: bool = False,
):
    data = state.data

    if data is None:
        raise ValueError("No data loaded.")

    eps = 1e-10

    if invert:
        if state.log_base != base:
            raise ValueError("Cannot invert log transformation with a different base.")

        data.iloc[:, 1:] = base ** data.iloc[:, 1:] - eps
        state.was_log_transformed = False
        state.log_base = None

    else:
        if state.was_log_transformed:
            raise ValueError("Data has already been log transformed.")

        data.iloc[:, 1:] = np.log(data.iloc[:, 1:] + eps) / np.log(base)
        state.was_log_transformed = True
        state.log_base = base

    state.data = data

    return {
        "log_transformed": state.was_log_transformed,
        "base": base,
    }


@register_operation(
    id="scaler_pareto",
    label="Pareto Scaling",
    description="Apply feature-wise Pareto scaling to the data.",
    citation="",
    category_tags=[OperationTag.TRANSFORMATION, OperationTag.SCALING],
    parameter_schema=[],
    requires=["data"],
    produces=["data", "was_scaled", "was_centered"],
)
def scaler_pareto(state: WorkflowState):
    """
    Apply feature-wise Pareto scaling.

    Old PySPRESSO logic:
        mean = feature mean
        std = feature standard deviation
        scaled = (x - mean) / sqrt(std)

    Pareto scaling includes mean-centering.
    """

    if state.data is None:
        raise ValueError("No data found. Run dataset initialization first.")

    data = state.data.copy()
    report = state.report

    X = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    if state.was_scaled:
        print(
            "Warning: Data has already been scaled using "
            + str(state.was_scaled)
            + ". This may lead to unexpected results."
        )

    mean = X.mean(axis=1)
    std = X.std(axis=1, ddof=1).replace(0, np.nan)

    data.iloc[:, 1:] = X.sub(mean, axis=0).div(np.sqrt(std), axis=0)

    state.data = data
    state.was_scaled = "pareto"
    state.was_centered = True

    if report is not None:
        report.add_together(
            [
                ("text", "Data were feature-wise Pareto scaled."),
                "line",
            ]
        )

    return {
        "message": "Data were feature-wise Pareto scaled.",
        "was_scaled": state.was_scaled,
        "was_centered": state.was_centered,
    }
