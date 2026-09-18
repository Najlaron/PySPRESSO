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

    base = float(base)
    if base <= 0 or base == 1:
     raise ValueError("Log base must be positive and different from 1.")

    values = data.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    if not invert and (values < 0).any().any():
        raise ValueError(
            "Log transformation cannot be applied because negative intensity values are present."
        )

    eps = 1e-10

    if invert:
        if not state.was_log_transformed:
            raise ValueError("Data are not currently log transformed.")

        if state.log_base != base:
            raise ValueError(
                "Cannot invert log transformation with a different base."
            )

        data.iloc[:, 1:] = base ** values - eps
        state.was_log_transformed = False
        state.log_base = None

    else:
        if state.was_log_transformed:
            raise ValueError("Data have already been log transformed.")

        data.iloc[:, 1:] = np.log(values + eps) / np.log(base)
        state.was_log_transformed = True
        state.log_base = base

    state.data = data

    # REPORTING ---------------------------------------------------------
    if invert:
        add_text(
            state,
            f"Log transformation with base {base} was inverted.",
            title="Inverse log transformation",
        )
    else:
        add_text(
            state,
            f"Sample intensities were log-transformed using base {base}.",
            title="Log transformation",
        )

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

    if state.was_scaled:
        raise ValueError(
            f"Data have already been scaled using '{state.was_scaled}'. "
            "Scaling the data again would produce a different transformation."
        )

    data = state.data.copy()

    try:
        X = data.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Pareto scaling requires numeric sample intensity values."
        ) from exc
    
    mean = X.mean(axis=1)
    std = X.std(axis=1, ddof=1)

    zero_variance_mask = std.eq(0) | std.isna()

    if zero_variance_mask.any():
        bad_features = data.loc[
            zero_variance_mask,
            data.columns[0],
        ].astype(str).tolist()

        raise ValueError(
            f"Pareto scaling cannot be applied to "
            f"{len(bad_features)} feature(s) with zero or undefined variance. "
            f"Examples: {bad_features[:10]}"
        )

    data.iloc[:, 1:] = X.sub(mean, axis=0).div(np.sqrt(std), axis=0)

    state.data = data
    state.was_scaled = "pareto"
    state.was_centered = True

    # REPORTING ---------------------------------------------------------
    add_text(
        state,
        (
            f"Data were feature-wise Pareto scaled. "
            f"Each feature was mean-centered and divided by the square root "
            f"of its standard deviation."
        ),
        title="Pareto scaling",
    )
    add_text(
        state,
        (
            f"Scaled features: {data.shape[0]}. "
            f"Samples per feature: {data.shape[1] - 1}."
        ),
    )

    return {
        "message": "Data were feature-wise Pareto scaled.",
        "was_scaled": state.was_scaled,
        "was_centered": state.was_centered,
    }
