import numpy as np

from core.registry import register_operation
from core.operation_models import OperationTag, ParameterDef
from core.workflow_models import WorkflowState


@register_operation(
    id="transformer_log",
    label="Log Transform",
    description="Apply or invert log transformation on sample intensity columns.",
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
            raise ValueError(
                "Cannot invert log transformation with a different base."
            )

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