from datetime import datetime
from pyspresso_app.core.registry import get_operation
from pyspresso_app.core.validation import validate_step
from pyspresso_app.core.workflow_models import Workflow, WorkflowStep, StepStatus
from pyspresso_app.core.html_reporter import (
    begin_step,
    finish_step,
)

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
    report_started = False

    # Start HTML report block
    try:
        begin_step(
            workflow.state,
            workflow_name=workflow.name,
            operation_name=operation.label,
            operation_id=operation.id,
            step_id=step.step_id,
            parameters=step.params,
        )

        report_started = True

    except Exception as report_exc:
        step.warnings.append(
            f"HTML report initialization failed: {report_exc}"
        )
    # Execute analytical operation
    try:
        result = operation.func(
            workflow.state,
            **step.params,
        )

        if isinstance(result, dict):
            step.output_summary = result
        else:
            step.output_summary = {
                "result": result
            }

        step.status = StepStatus.DONE
        step.messages.append(
            "Step completed successfully."
        )
        # Finish HTML report block
        if report_started:
            try:
                finish_step(
                    workflow.state,
                    status="success",
                )

            except Exception as report_exc:
                step.warnings.append(
                    f"HTML report finalization failed: {report_exc}"
                )
        # Execution log
        workflow.state.execution_log.append(
            {
                "step_id": step.step_id,
                "operation_id": step.operation_id,
                "status": step.status.value,
                "time": datetime.now().isoformat(),
                "summary": step.output_summary,
                "report_html_path": getattr(
                    workflow.state,
                    "report_html_path",
                    None,
                ),
            }
        )
    # Analytical operation failed
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.valid = False
        step.messages.append(str(exc))

        # Mark report block as failed.
        if report_started:
            try:
                finish_step(
                    workflow.state,
                    status="error",
                )

            except Exception as report_exc:
                step.warnings.append(
                    f"HTML report finalization failed: {report_exc}"
                )

        workflow.state.execution_log.append(
            {
                "step_id": step.step_id,
                "operation_id": step.operation_id,
                "status": step.status.value,
                "time": datetime.now().isoformat(),
                "error": str(exc),
                "report_html_path": getattr(
                    workflow.state,
                    "report_html_path",
                    None,
                ),
            }
        )

    return step

def run_workflow(workflow: Workflow) -> Workflow:
    for step in workflow.steps:
        run_step(
            workflow,
            step,
        )

        if step.status == StepStatus.FAILED:
            break

    return workflow