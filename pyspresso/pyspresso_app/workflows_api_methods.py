import uuid
import os
from pathlib import Path

from flask import request, jsonify
from werkzeug.utils import secure_filename
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
import pandas as pd
import math

# místo, kam se ukládáají data
UPLOAD_FOLDER = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
# povolené formáty dat
ALLOWED_EXTENSIONS = {"csv", "txt", "xlsx", "xls", "tsv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, subfolder="workflows"):
    """Uloží nahraný soubor a vrátí cestu."""
    if not file or file.filename == "":
        return None

    if not allowed_file(file.filename):
        return None

    # vytvoří složku pro dané workflow
    folder_path = UPLOAD_FOLDER / subfolder
    folder_path.mkdir(exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = folder_path / filename

    file.save(str(filepath))
    return str(filepath.relative_to(UPLOAD_FOLDER.parent))


# aktualizuje záznam v databázi z instance třídy workflow
def save_workflow(workflow_id: str, workflow: Workflow):
    workflow_row = WorkflowORM.query.get(workflow_id)
    if not workflow_row:
        return False

    def _sanitize_for_json(obj):
        """Recursively replace NaN/Infinity with None so JSON is valid.

        Uses pandas.isna to catch pandas/numpy NA types and math for floats.
        """
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None

        return obj

    workflow_row.definition = _sanitize_for_json(workflow.definition.to_dict())
    workflow_row.state = _sanitize_for_json(workflow.state.to_dict())
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
        operation = get_operation(operation_id)
        return operation.func
    except KeyError:
        return None


# API Endpoints
# Vytvoří nové workflow a uloží ho do databáze
@app.route("/new_workflow", methods=["POST"])
def create_new_workflow():
    workflow_name = request.form.get("workflowName", "").strip()
    folder_name = request.form.get("folderName", "").strip()
    report_file_name = request.form.get("reportFileName", "").strip()

    # kontrola, jestli byly vyplněné povinné pole
    if not workflow_name:
        return jsonify({"message": "workflowName is required."}), 400

    if not folder_name:
        return jsonify({"message": "folderName is required."}), 400

    if not report_file_name:
        return jsonify({"message": "reportFileName is required."}), 400

    # uloží data a batch info a vratí cesty k nim (možná hodit do samotné funkce, at tady toho není moc)
    files_dict = {}
    if "data" in request.files:
        file = request.files["data"]
        if file and file.filename:
            filepath = save_uploaded_file(file, folder_name or "workflows")
            if filepath:
                files_dict["data"] = filepath

    if "batchInfo" in request.files:
        file = request.files["batchInfo"]
        if file and file.filename:
            filepath = save_uploaded_file(file, folder_name or "workflows")
            if filepath:
                files_dict["batch_info"] = filepath

    workflow_id = str(uuid.uuid4())
    workflow = Workflow(workflow_id=workflow_id, name=workflow_name)

    # uloží cesty k souborům
    workflow.state.files = files_dict

    definition = workflow.definition.to_dict()
    state = workflow.state.to_dict()

    # sanitize to ensure no NaN/Inf remain before storing
    def _sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    definition = _sanitize_for_json(definition)
    state = _sanitize_for_json(state)

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
            }
        ),
        201,
    )


@app.route("/workflow/<workflow_id>/delete_step/<step_id>", methods=["DELETE"])
def delete_step(workflow_id: str, step_id: str):
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

    workflow.definition.steps.remove(step)
    save_workflow(workflow_id, workflow)

    return jsonify({"message": "Step deleted"}), 200


@app.route("/workflow/<workflow_id>/step/<step_id>/parameters", methods=["PUT"])
def update_step_parameters(workflow_id: str, step_id: str):
    """Update parameters for a workflow step."""
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

    payload = request.get_json(silent=True) or {}
    parameters = payload.get("parameters", {})

    if not isinstance(parameters, dict):
        return jsonify({"message": "Parameters must be a dictionary"}), 400

    step.params = parameters

    save_workflow(workflow_id, workflow)

    return (
        jsonify(
            {
                "message": "Step parameters updated",
            }
        ),
        200,
    )


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
                        "example": param.example,
                    }
                    for param in op.parameter_schema
                ],
                "requires": op.requires,
                "produces": op.produces,
            }
        )

    return jsonify(operations_data), 200


@app.route("/workflow/<workflow_id>/step/<step_id>/run", methods=["POST"])
def execute_step(workflow_id: str, step_id: str):
    """Execute a single workflow step."""
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

    try:
        step = run_step(workflow, step)
        save_workflow(workflow_id, workflow)

        return (
            jsonify(
                {
                    "message": "Step executed",
                    # "step": step.to_dict(),
                    # "workflow_state": workflow.state.to_dict(),
                }
            ),
            200,
        )
    except Exception as ex:
        return jsonify({"message": str(ex), "step": step.to_dict()}), 400


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
