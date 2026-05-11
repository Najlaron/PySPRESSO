from typing import Any
from core.workflow_models import WorkflowState, WorkflowStep, StepStatus
from core.operation_models import OperationDefinition


def validate_step(
    state: WorkflowState,
    step: WorkflowStep,
    operation: OperationDefinition,
) -> WorkflowStep:
    step.messages.clear()
    step.warnings.clear()
    step.valid = True

    missing_params = []
    for param in operation.parameter_schema:
        if param.required and step.params.get(param.name) is None:
            missing_params.append(param.name)

    if missing_params:
        step.valid = False
        step.status = StepStatus.NEEDS_PARAMETERS
        step.messages.append(
            "Missing parameters: " + ", ".join(missing_params)
        )
        return step

    missing_requirements = []
    for requirement in operation.requires:
        value = getattr(state, requirement, None)
        if value is None:
            missing_requirements.append(requirement)

    if missing_requirements:
        step.valid = False
        step.status = StepStatus.BLOCKED
        step.messages.append(
            "Missing required workflow state: " + ", ".join(missing_requirements)
        )
        return step

    step.status = StepStatus.READY
    return step