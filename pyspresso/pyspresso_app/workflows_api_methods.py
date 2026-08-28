import shutil
import uuid
import os
from pathlib import Path
import json
from datetime import datetime

from flask import request, jsonify, send_from_directory, abort
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
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
# povolené formáty dat
ALLOWED_EXTENSIONS = {"csv", "txt", "xlsx", "xls", "tsv"}

# místo, kde jsou obrázky a další vytvořené soubory
OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# vytvoří databázovou tabulku, pokud ještě nění vytvořena
def ensure_database_tables() -> None:
    with app.app_context():
        db.create_all()


ensure_database_tables()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, subfolder="workflows"):
    if not file or file.filename == "":
        return None

    # vytvoří soubor pouze pokud jsou data v povoleném formátu
    if not allowed_file(file.filename):
        return None

    # vytvoří složku pro dané workflow
    folder_path = UPLOAD_FOLDER / subfolder
    print(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = folder_path / filename

    file.save(str(filepath))
    return str(filepath.relative_to(UPLOAD_FOLDER.parent))


# aktualizuje záznam v databázi z instance třídy workflow
def save_workflow(workflow_id: str, workflow: Workflow):
    workflow_row = db.session.get(WorkflowORM, workflow_id)
    if not workflow_row:
        return False

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

    workflow_row.definition = _sanitize_for_json(workflow.definition.to_dict())
    workflow_row.state = _sanitize_for_json(workflow.state.to_dict())
    db.session.commit()
    return True


# vytvoří instanci workflow ze záznamu z databáze
def load_workflow(workflow_id: str):
    workflow_row = db.session.get(WorkflowORM, workflow_id)
    if not workflow_row:
        return None

    workflow = Workflow(workflow_id=workflow_id, name=workflow_row.workflow_name)
    workflow.definition = WorkflowDefinition.from_dict(workflow_row.definition)
    workflow.state = WorkflowState.from_dict(workflow_row.state)
    return workflow


# přidá inicialiační metodu podle zvoleného formátu dat do nově vytvořeného workflow
def add_init_step(wf, data_format: str):
    format_to_initializer = {
        "cd": "initializer_compound_discoverer",
    }

    selected_format = (data_format or "").strip().lower()
    operation_id = format_to_initializer.get(selected_format)
    if not operation_id:
        supported_formats = ", ".join(sorted(format_to_initializer.keys()))
        raise ValueError(
            f"Unsupported dataFormat '{data_format}'. "
            f"Supported values: {supported_formats}"
        )

    try:
        get_operation(operation_id)
    except KeyError as ex:
        raise KeyError(f"Operation '{operation_id}' not found") from ex

    step_id = str(uuid.uuid4())
    new_step = WorkflowStep(
        step_id=step_id,
        operation_id=operation_id,
    )

    wf.definition.steps.append(new_step)


# vrátí konkrétní metodu
def get_operation_func(operation_id: str):
    try:
        operation = get_operation(operation_id)
        return operation.func
    except KeyError:
        return None


########################################
# API Endpoints
########################################


# vytvoří nové workflow a uloží ho do databáze
@app.route("/new_workflow", methods=["POST"])
def create_new_workflow():
    workflow_name = request.form.get("workflowName", "").strip()
    folder_name = request.form.get("folderName", "").strip()
    report_file_name = request.form.get("reportFileName", "").strip()
    data_format = request.form.get("dataFormat", "").strip()

    # kontrola, jestli byly vyplněné povinné pole
    if not workflow_name:
        return jsonify({"message": "workflowName is required."}), 400

    if not folder_name:
        return jsonify({"message": "folderName is required."}), 400

    # if not report_file_name:
    #     return jsonify({"message": "reportFileName is required."}), 400

    if not data_format:
        return jsonify({"message": "dataFormat is required."}), 400

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
    workflow.state.main_folder = folder_name

    # přidání inicializačního kroku pro data
    # add_init_step(workflow)

    # import kroků z jiného workflow, pokud není nahraný, přidá se inicializační metoda
    if "importFile" in request.files:
        import_file = request.files["importFile"]
        if import_file and import_file.filename:
            try:
                _ = import_methods_from_file(workflow, import_file)
            except ValueError as ex:
                return jsonify({"message": str(ex)}), 400
            except KeyError as ex:
                return jsonify({"message": str(ex)}), 400
    else:
        # přidání inicializačního kroku pro data
        try:
            add_init_step(workflow, data_format)
        except ValueError as ex:
            return jsonify({"message": str(ex)}), 400
        except KeyError as ex:
            return jsonify({"message": str(ex)}), 404

    # uloží cesty k souborům
    workflow.state.files = files_dict

    definition = workflow.definition.to_dict()
    state = workflow.state.to_dict()

    # před uložením nezůstanou žádné hodnoty NaN ani Inf
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


# vrátí workflow z databáze podle ID
@app.route("/workflow/id/<workflow_id>", methods=["GET"])
def get_workflow(workflow_id: str):
    workflow_row = db.session.get(WorkflowORM, workflow_id)

    if not workflow_row:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    return jsonify(workflow_row.to_dict()), 200


# aktualizuje popis u workflow
@app.route("/workflow/<workflow_id>/description", methods=["POST"])
def update_description(workflow_id: str):
    workflow_row = db.session.get(WorkflowORM, workflow_id)

    if not workflow_row:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    payload = request.get_json(silent=True) or {}

    if "description" not in payload:
        return jsonify({"message": "Description is required."}), 400

    description = payload.get("description")

    if description is not None and not isinstance(description, str):
        return jsonify({"message": "Description must be a string."}), 400

    description = description.strip()
    workflow_row.description = description

    try:
        db.session.commit()
    except Exception as ex:
        db.session.rollback()
        return jsonify({"message": str(ex)}), 400

    return (
        jsonify(
            {
                "message": "Workflow description updated.",
                "workflowId": workflow_row.id,
                "description": workflow_row.description,
            }
        ),
        200,
    )


# přidání kroku do workflow
@app.route("/workflow/<workflow_id>/step", methods=["POST"])
def add_workflow_step(workflow_id: str):
    # vytvoří třídu workflow ze záznamu z databáze
    workflow = load_workflow(workflow_id)
    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

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


# smaže workflow
@app.route("/workflow/<workflow_id>/delete", methods=["DELETE"])
def delete_workflow(workflow_id: str):
    workflow_row = db.session.get(WorkflowORM, workflow_id)

    if not workflow_row:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    workflow_folder_name = workflow_row.folder_name

    try:
        workflow_folder = OUTPUT_FOLDER / f"{workflow_folder_name}"

        # s workflow se pokusí smazat i příslušnou složku
        if workflow_folder.exists():
            shutil.rmtree(workflow_folder)

        db.session.delete(workflow_row)
        db.session.commit()
    except Exception as ex:
        db.session.rollback()
        return jsonify({"message": str(ex)}), 400

    return (
        jsonify({"message": "Workflow was deleted."}),
        200,
    )


# @app.route("/workflow/<workflow_id>/folder", methods=["GET"])
# def open_workflow_folder(workflow_id: str):
#     workflow_row = load_workflow(workflow_id)

#     if not workflow_row:
#         return (
#             jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
#             404,
#         )

#     workflow_folder = Path(workflow_row.state.main_folder)

#     if workflow_folder.exists():
#         os.startfile(workflow_folder)
#         return (
#             jsonify({"message": "Workflow folder was opened."}),
#             200,
#         )
#     else:
#         return (
#             jsonify({"message": "Workflow folder does not exist"}),
#             400,
#         )


# smazání kroku
@app.route("/workflow/<workflow_id>/delete_step/<step_id>", methods=["DELETE"])
def delete_step(workflow_id: str, step_id: str):
    workflow = load_workflow(workflow_id)
    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    # najde konrétní krok podle jeho ID
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


# nastavení parametrů metody
@app.route("/workflow/<workflow_id>/step/<step_id>/parameters", methods=["PUT"])
def update_step_parameters(workflow_id: str, step_id: str):
    workflow = load_workflow(workflow_id)
    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

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


# přeuspořádání kroků
@app.route("/workflow/<workflow_id>/reorder", methods=["PUT"])
def reorder_workflow_steps(workflow_id: str):
    workflow = load_workflow(workflow_id)
    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    payload = request.get_json(silent=True) or {}
    step_ids = payload.get("stepIds", [])

    if not isinstance(step_ids, list) or not step_ids:
        return jsonify({"message": "stepIds must be a non-empty array"}), 400

    step_map = {step.step_id: step for step in workflow.definition.steps}
    missing_ids = [step_id for step_id in step_ids if step_id not in step_map]
    if missing_ids:
        return jsonify({"message": f"Unknown step ids: {missing_ids}"}), 400

    reordered_steps = [step_map[step_id] for step_id in step_ids]
    workflow.definition.steps = reordered_steps
    save_workflow(workflow_id, workflow)

    return jsonify({"message": "Workflow steps reordered"}), 200


# vrátí všechny workflow z databáze
@app.route("/workflows", methods=["GET"])
def get_workflows():
    saved_workflows = WorkflowORM.query.all()
    return jsonify([w.to_dict() for w in saved_workflows]), 200


# vrátí všechny operace
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


# vykoná konkrétní krok (metodu)
@app.route("/workflow/<workflow_id>/step/<step_id>/run", methods=["POST"])
def execute_step(workflow_id: str, step_id: str):
    workflow = load_workflow(workflow_id)
    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

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
                    "stepMessage": step.messages,
                    "stepOperationId": step.operation_id,
                    "stepStatus": step.status,
                }
            ),
            200,
        )
    except Exception as ex:
        return jsonify({"message": str(ex), "step": step.to_dict()}), 400


# vrátí obrázek podle cesty
@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output_file(filename):
    output_root = OUTPUT_FOLDER.resolve()
    requested_path = (output_root / filename).resolve()

    try:
        requested_path.relative_to(output_root)
    except ValueError:
        abort(403)

    if not requested_path.is_file():
        return jsonify({"message": "Output file not found"}), 404

    return send_from_directory(output_root, filename)


# export metod i s jejich aktuálně nastavenými parametry
@app.route("/workflow/<workflow_id>/export", methods=["GET"])
def export_workflow(workflow_id: str):
    workflow = load_workflow(workflow_id)

    if not workflow:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    # ops = [{"operation_id": step.operation_id} for step in workflow.definition.steps]
    ops = []
    for step in workflow.definition.steps:
        # if step.operation_id == "initializer_compound_discoverer":
        #     continue
        # else:
        #     ops.append({"operation_id": step.operation_id, "params": step.params})
        ops.append({"operation_id": step.operation_id, "params": step.params})

    payload = {"operations": ops}
    resp = jsonify(payload)
    resp.headers["Content-Disposition"] = (
        f"attachment; filename=workflow_{workflow_id}.json"
    )
    return resp, 200


# pomocná metoda pro import kroků, přidá kroky i s paramatry do workflow
def import_methods_from_file(workflow: Workflow, import_file):
    if not import_file or getattr(import_file, "filename", "") == "":
        raise ValueError("Uploaded import file is empty.")

    try:
        data = json.loads(import_file.read())
    except Exception as ex:
        raise ValueError(f"Invalid import JSON file: {ex}")

    ops = data.get("operations") or []
    if not isinstance(ops, list):
        raise ValueError("Invalid format: 'operations' must be a list.")

    created_steps = []

    for item in ops:
        if isinstance(item, dict):
            op_id = item.get("operation_id")
            op_params = item.get("params")
        else:
            raise ValueError("Invalid format: one operation must be a dictionary.")

        if not op_id:
            continue
        elif op_params is None:
            op_params = {}

        try:
            get_operation(op_id)
        except KeyError:
            raise KeyError(f"Operation {op_id} does not exist.")

        step_id = str(uuid.uuid4())
        new_step = WorkflowStep(step_id=step_id, operation_id=op_id, params=op_params)
        workflow.definition.steps.append(new_step)
        created_steps.append({"step_id": step_id, "operation_id": op_id})

    return {"created_steps": created_steps}


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
