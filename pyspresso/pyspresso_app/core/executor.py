from datetime import datetime
from typing import Any

from pyspresso_app.core.registry import get_operation
from pyspresso_app.core.validation import validate_step
from pyspresso_app.core.workflow_models import Workflow, WorkflowStep, StepStatus


def run_step(workflow: Workflow, step: WorkflowStep) -> WorkflowStep:
    operation = get_operation(step.operation_id)

    validate_step(workflow.state, step, operation)

    if not step.enabled:
        step.status = StepStatus.NOT_RUN
        step.messages.append("Step is disabled.")
        return step

    if not step.valid:
        return step

    step.status = StepStatus.RUNNING

    try:
        result = operation.func(workflow.state, **step.params)

        if isinstance(result, dict):
            step.output_summary = result
        else:
            step.output_summary = {"result": result}

        step.status = StepStatus.DONE
        step.messages.append("Step completed.")

        workflow.state.execution_log.append(
            {
                "step_id": step.step_id,
                "operation_id": step.operation_id,
                "status": step.status.value,
                "time": datetime.now().isoformat(),
                "summary": step.output_summary,
            }
        )

    except Exception as exc:
        step.status = StepStatus.FAILED
        step.valid = False
        step.messages.append(str(exc))

        workflow.state.execution_log.append(
            {
                "step_id": step.step_id,
                "operation_id": step.operation_id,
                "status": step.status.value,
                "time": datetime.now().isoformat(),
                "error": str(exc),
            }
        )

    return step


def run_workflow(workflow: Workflow) -> Workflow:
    for step in workflow.steps:
        run_step(workflow, step)

        if step.status == StepStatus.FAILED:
            break

    return workflow
