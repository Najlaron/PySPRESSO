import uuid

from flask import request, jsonify
from pyspresso_app.config import app, db
from pyspresso_app.core.workflow_models import (
    WorkflowORM,
    Workflow,
    WorkflowState,
    WorkflowDefinition,
    WorkflowStep,
)
from pyspresso_app.core.registry import get_operation, list_operations
from pyspresso_app.core.executor import run_step

# def load_workflow_state(workflow_id: str):
#     workflow_row = WorkflowORM.query.get(workflow_id)
#     if not workflow_row:
#         return None
#     return WorkflowState.from_dict(workflow_row.state)


# def save_workflow_state(workflow_id: str, state: WorkflowState):
#     workflow_row = WorkflowORM.query.get(workflow_id)
#     if not workflow_row:
#         return False

#     workflow_row.state = state.to_dict()
#     db.session.commit()
#     return True


# Uloží záznam do databáze z instance třídy workflow workflow
def save_workflow(workflow_id: str, workflow: Workflow):
    workflow_row = WorkflowORM.query.get(workflow_id)
    if not workflow_row:
        return False

    workflow_row.definition = workflow.definition.to_dict()
    workflow_row.state = workflow.state.to_dict()
    db.session.commit()
    return True


# Vytvoří instanci workflow ze záznamu z databáze
def load_workflow(workflow_id: str):
    workflow_row = WorkflowORM.query.get(workflow_id)
    if not workflow_row:
        return None

    workflow = Workflow(workflow_id=workflow_id, name=workflow_row.workflow_name)
    workflow.definition = WorkflowDefinition.from_dict(workflow_row.definition)
    workflow.state = WorkflowState.from_dict(workflow_row.state)
    return workflow


def get_operation_func(operation_id: str):
    try:
        operation_def = get_operation(operation_id)
        return operation_def.func
    except KeyError:
        return None


# API Endpoints
# Vytvoří nové workflow a uloží ho do databáze
@app.route("/new_workflow", methods=["POST"])
def create_new_workflow():
    payload = request.get_json(silent=True) or {}

    workflow_name = payload.get("workflowName", "").strip()
    folder_name = payload.get("folderName", "").strip() or None
    report_file_name = payload.get("reportFileName", "").strip() or None

    if not workflow_name:
        return jsonify({"message": "workflowName is required."}), 400

    workflow_id = str(uuid.uuid4())
    workflow = Workflow(workflow_id=workflow_id, name=workflow_name)

    definition = workflow.definition.to_dict()
    state = workflow.state.to_dict()

    workflow_row = WorkflowORM(
        id=workflow_id,
        workflow_name=workflow_name,
        pyspresso_version=workflow.pyspresso_version,
        definition=definition,
        state=state,
        folder_name=folder_name,
        report_file_name=report_file_name,
    )

    try:
        db.session.add(workflow_row)
        db.session.commit()
    except Exception as ex:
        return jsonify({"message": str(ex)}), 400

    return (
        jsonify(
            {
                "message": "Workflow was created",
                "workflowId": workflow_id,
            }
        ),
        201,
    )


# Vráti workflow z databáze podle ID
@app.route("/workflow/<workflow_id>", methods=["GET"])
def get_workflow(workflow_id: str):
    workflow_row = WorkflowORM.query.get(workflow_id)

    if not workflow_row:
        return jsonify({"message": "Workflow not found"}), 404

    return jsonify(workflow_row.to_dict()), 200


# Přidá krok do workflow
@app.route("/workflow/<workflow_id>/step", methods=["POST"])
def add_workflow_step(workflow_id: str):
    # vytvoří třídu workflow ze záznamu z databáze
    workflow = load_workflow(workflow_id)
    if not workflow:
        return jsonify({"message": "Workflow not found"}), 404

    payload = request.get_json(silent=True) or {}
    operation_id = payload.get("operationId", "").strip()
    params = payload.get("params", {})

    if not operation_id:
        return jsonify({"message": "operationId is required"}), 400

    # kontrola, že operace existuje
    try:
        get_operation(operation_id)
    except KeyError:
        return jsonify({"message": f"Operation '{operation_id}' not found"}), 404

    step_id = str(uuid.uuid4())
    new_step = WorkflowStep(
        step_id=step_id,
        operation_id=operation_id,
        params=params,
    )

    workflow.definition.steps.append(new_step)
    save_workflow(workflow_id, workflow)

    return (
        jsonify(
            {
                "message": "Step added",
                # "step": new_step.to_dict(),
                # "definition": workflow.definition.to_dict(),
            }
        ),
        201,
    )


@app.route("/workflow/<workflow_id>/delete_step/<step_id>", methods=["DELETE"])
def delete_step(workflow_id: str, step_id: str):
    print(
        "delete_step called:",
        request.method,
        request.path,
        "Origin:",
        request.headers.get("Origin"),
    )
    workflow = load_workflow(workflow_id)
    if not workflow:
        return jsonify({"message": "Workflow not found"}), 404

    step = None
    for s in workflow.definition.steps:
        if s.step_id == step_id:
            step = s
            break

    if not step:
        return jsonify({"message": f"Step '{step_id}' not found"}), 404

    print("zdarec")
    workflow.definition.steps.remove(step)
    print("zdarec")
    save_workflow(workflow_id, workflow)

    return jsonify({"message": "Step deleted"}), 200


# @app.route("/operation/<operation_id>", methods=["GET"])
# def get_operation_detail(operation_id):
#     """Return details of a specific operation."""
#     try:
#         op = get_operation(operation_id)
#     except KeyError:
#         return jsonify({"message": f"Operation '{operation_id}' not found"}), 404

#     operation_data = {
#         "id": op.id,
#         "label": op.label,
#         "description": op.description,
#         "categoryTags": [tag.value for tag in op.category_tags],
#         "parameterSchema": [
#             {
#                 "name": param.name,
#                 "type": param.type,
#                 "required": param.required,
#                 "default": param.default,
#                 "label": param.label,
#                 "help": param.help,
#             }
#             for param in op.parameter_schema
#         ],
#         "requires": op.requires,
#         "produces": op.produces,
#     }

#     return jsonify(operation_data), 200


# Vrátí všechny workflow z databáze
@app.route("/workflows", methods=["GET"])
def get_workflows():
    saved_workflows = WorkflowORM.query.all()
    return jsonify([w.to_dict() for w in saved_workflows]), 200


# Vrátí všechny operace
@app.route("/operations", methods=["GET"])
def get_available_operations():
    operations = list_operations()

    operations_data = []
    for op in operations:
        operations_data.append(
            {
                "id": op.id,
                "label": op.label,
                "description": op.description,
                "categoryTags": [tag.value for tag in op.category_tags],
                "parameterSchema": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "required": param.required,
                        "default": param.default,
                        "label": param.label,
                        "help": param.help,
                    }
                    for param in op.parameter_schema
                ],
                "requires": op.requires,
                "produces": op.produces,
            }
        )

    return jsonify(operations_data), 200


# @app.route("/workflow/<workflow_id>/step/<step_id>/run", methods=["POST"])
# def execute_step(workflow_id: str, step_id: str):
#     """Execute a single workflow step."""
#     workflow = load_workflow(workflow_id)
#     if not workflow:
#         return jsonify({"message": "Workflow not found"}), 404

#     step = None
#     for s in workflow.definition.steps:
#         if s.step_id == step_id:
#             step = s
#             break

#     if not step:
#         return jsonify({"message": f"Step '{step_id}' not found"}), 404

#     payload = request.get_json(silent=True) or {}
#     if "params" in payload:
#         step.params.update(payload["params"])

#     # Run the step
#     try:
#         step = run_step(workflow, step)
#         save_workflow(workflow_id, workflow)

#         return (
#             jsonify(
#                 {
#                     "message": "Step executed",
#                     "step": step.to_dict(),
#                     "workflow_state": workflow.state.to_dict(),
#                 }
#             ),
#             200,
#         )
#     except Exception as ex:
#         return jsonify({"message": str(ex), "step": step.to_dict()}), 400


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
