import shutil
import uuid
from pathlib import Path
import json
import math
import copy

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
from pyspresso_app.bootstrap import initialize
from pyspresso_app.core.html_reporter import get_report_path

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


initialize()
ensure_database_tables()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, subfolder="workflows"):
    if not file or file.filename == "":
        return None

    # vytvoří soubor pouze pokud jsou data v povoleném formátu
    if not allowed_file(file.filename):
        return None

    # Never allow a client-provided folder name to escape the uploads directory.
    subfolder = _validate_folder_name(subfolder)
    folder_path = UPLOAD_FOLDER / subfolder
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = folder_path / filename

    file.save(str(filepath))
    return str(filepath.relative_to(UPLOAD_FOLDER.parent))

def _validate_folder_name(folder_name: str) -> str:
    folder_name = str(folder_name or "").strip()
    candidate = Path(folder_name)
    if (
        not folder_name
        or candidate.is_absolute()
        or candidate.name != folder_name
        or folder_name in {".", ".."}
        or "/" in folder_name
        or "\\" in folder_name
    ):
        raise ValueError(
            "folderName must be a single folder name without path separators."
        )
    return folder_name

def _json_safe_for_database(value):
    """
    Convert workflow data to values accepted by the database JSON column.

    This is persistence logic only.
    It has nothing to do with HTML reporting.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe_for_database(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe_for_database(item)
            for item in value
        ]

    # numpy scalar types
    if hasattr(value, "item"):
        try:
            return _json_safe_for_database(value.item())
        except Exception:
            pass

    # datetime / pandas Timestamp / similar
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    raise TypeError(
        f"Value of type {type(value).__name__} "
        "cannot be stored in workflow JSON."
    )

# aktualizuje záznam v databázi z instance třídy workflow
def save_workflow(workflow_id: str, workflow: Workflow):
    workflow_row = WorkflowORM.query.filter_by(
        id=workflow_id
    ).first()

    if not workflow_row:
        return False

    workflow_row.definition = _json_safe_for_database(
        workflow.definition.to_dict()
    )
    workflow_row.state = _json_safe_for_database(
        workflow.state.to_dict()
    )

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


def load_workflow_definition(workflow_id: str):
    """Load only the small workflow definition, leaving analytical state deferred."""
    result = (
        db.session.query(WorkflowORM.definition)
        .filter(WorkflowORM.id == workflow_id)
        .first()
    )
    if result is None:
        return None
    return WorkflowDefinition.from_dict(result[0])


def save_workflow_definition(workflow_id: str, definition: WorkflowDefinition):
    """Persist an editor-only change without reading or rewriting full table state."""

    definition_payload = _json_safe_for_database(
        definition.to_dict()
    )
    updated_rows = (
        WorkflowORM.query
        .filter_by(id=workflow_id)
        .update(
            {"definition": definition_payload},
            synchronize_session=False,
        )
    )
    if updated_rows == 0:
        return None

    db.session.commit()

    return definition_payload

def _get_default_operation_params(operation_id: str) -> dict:
    """
    Build the complete default parameter dictionary for an operation.

    A deep copy is used so mutable defaults such as lists are not shared
    between workflow steps.
    """
    operation = get_operation(operation_id)

    return {
        parameter.name: copy.deepcopy(parameter.default)
        for parameter in operation.parameter_schema
    }

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
        default_params = _get_default_operation_params(operation_id)

    except KeyError as ex:
        raise KeyError(
            f"Operation '{operation_id}' not found"
        ) from ex

    step_id = str(uuid.uuid4())

    new_step = WorkflowStep(
        step_id=step_id,
        operation_id=operation_id,
        params=default_params,
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
    if not report_file_name:
        report_file_name = "report"

    # kontrola, jestli byly vyplněné povinné pole
    if not workflow_name:
        return jsonify({"message": "workflowName is required."}), 400

    if not folder_name:
        return jsonify({"message": "folderName is required."}), 400

    try:
        folder_name = _validate_folder_name(folder_name)
    except ValueError as ex:
        return jsonify({"message": str(ex)}), 400

    # if WorkflowORM.query.filter_by(folder_name=folder_name).first() is not None:
    #     return jsonify(
    #         {"message": f"A workflow using folder '{folder_name}' already exists."}
    #     ), 409

    data_format = request.form.get("dataFormat", "").strip()
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
    workflow.state.report_file_name = report_file_name or "report"

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

    definition = _json_safe_for_database(
        workflow.definition.to_dict()
    )

    state = _json_safe_for_database(
        workflow.state.to_dict()
    )

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
    definition = load_workflow_definition(workflow_id)

    if not definition:
        return (
            jsonify(
                {
                    "message":
                    f"Workflow with ID:'{workflow_id}' does not exist."
                }
            ),
            404,
        )

    payload = request.get_json(silent=True) or {}

    operation_id = payload.get("operationId", "").strip()
    submitted_params = payload.get("params", {})

    if not operation_id:
        return jsonify(
            {"message": "operationId is required"}
        ), 400

    if not isinstance(submitted_params, dict):
        return jsonify(
            {"message": "params must be a dictionary"}
        ), 400

    # Check that the operation exists and obtain its defaults.
    try:
        default_params = _get_default_operation_params(
            operation_id
        )
    except KeyError:
        return jsonify(
            {
                "message":
                f"Operation '{operation_id}' not found"
            }
        ), 404

    # Start with all defaults.
    # Any values explicitly supplied by the frontend override them.
    params = default_params.copy()
    params.update(submitted_params)

    step_id = str(uuid.uuid4())

    new_step = WorkflowStep(
        step_id=step_id,
        operation_id=operation_id,
        params=params,
    )

    definition.steps.append(new_step)

    definition_payload = save_workflow_definition(workflow_id,definition,)

    return (
        jsonify(
            {
                "message": "Step added",
                "stepId": step_id,
                "definition": definition_payload,
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
        workflow_folder = (OUTPUT_FOLDER / f"{workflow_folder_name}").resolve()
        workflow_folder.relative_to(OUTPUT_FOLDER.resolve())
        upload_folder = (UPLOAD_FOLDER / f"{workflow_folder_name}").resolve()
        upload_folder.relative_to(UPLOAD_FOLDER.resolve())

        # s workflow se pokusí smazat i příslušnou složku
        if workflow_folder.exists():
            shutil.rmtree(workflow_folder)

        if upload_folder.exists():
            shutil.rmtree(upload_folder)

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
    definition = load_workflow_definition(workflow_id)
    if not definition:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    # najde konrétní krok podle jeho ID
    step = None
    for s in definition.steps:
        if s.step_id == step_id:
            step = s
            break

    if not step:
        return jsonify({"message": f"Step '{step_id}' not found"}), 404

    definition.steps.remove(step)
    definition_payload = save_workflow_definition(workflow_id, definition)

    return jsonify(
        {"message": "Step deleted", "definition": definition_payload}
    ), 200


# nastavení parametrů metody
@app.route("/workflow/<workflow_id>/step/<step_id>/parameters", methods=["PUT"])
def update_step_parameters(workflow_id: str, step_id: str):
    definition = load_workflow_definition(workflow_id)
    if not definition:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    step = None
    for s in definition.steps:
        if s.step_id == step_id:
            step = s
            break

    if not step:
        return jsonify({"message": f"Step '{step_id}' not found"}), 404

    payload = request.get_json(silent=True) or {}
    parameters = payload.get("parameters", {})

    if not isinstance(parameters, dict):
        return jsonify({"message": "Parameters must be a dictionary"}), 400

    default_params = _get_default_operation_params(step.operation_id)

    updated_params = default_params.copy()

    # Preserve previously stored values.
    updated_params.update(step.params or {})

    # Apply the newly submitted values.
    updated_params.update(parameters)

    step.params = updated_params

    definition_payload = save_workflow_definition(workflow_id, definition)

    return (
        jsonify(
            {
                "message": "Step parameters updated",
                "definition": definition_payload,
            }
        ),
        200,
    )


# přeuspořádání kroků
@app.route("/workflow/<workflow_id>/reorder", methods=["PUT"])
def reorder_workflow_steps(workflow_id: str):
    definition = load_workflow_definition(workflow_id)
    if not definition:
        return (
            jsonify({"message": f"Workflow with ID:'{workflow_id}' does not exist."}),
            404,
        )

    payload = request.get_json(silent=True) or {}
    step_ids = payload.get("stepIds", [])

    if not isinstance(step_ids, list) or not step_ids:
        return jsonify({"message": "stepIds must be a non-empty array"}), 400

    step_map = {step.step_id: step for step in definition.steps}
    missing_ids = [step_id for step_id in step_ids if step_id not in step_map]
    if missing_ids:
        return jsonify({"message": f"Unknown step ids: {missing_ids}"}), 400

    reordered_steps = [step_map[step_id] for step_id in step_ids]
    definition.steps = reordered_steps
    definition_payload = save_workflow_definition(workflow_id, definition)

    return jsonify(
        {"message": "Workflow steps reordered", "definition": definition_payload}
    ), 200


# vrátí všechny workflow z databáze
@app.route("/workflows", methods=["GET"])
def get_workflows():
    if request.args.get("summary", "").strip().lower() in {"1", "true", "yes"}:
        rows = db.session.query(
            WorkflowORM.id,
            WorkflowORM.workflow_name,
            WorkflowORM.pyspresso_version,
            WorkflowORM.folder_name,
            WorkflowORM.report_file_name,
            WorkflowORM.description,
            WorkflowORM.created_at,
            WorkflowORM.updated_at,
        ).all()
        return jsonify(
            [
                {
                    "id": row.id,
                    "workflow_name": row.workflow_name,
                    "pyspresso_version": row.pyspresso_version,
                    "folder_name": row.folder_name,
                    "report_file_name": row.report_file_name,
                    "description": row.description,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
                for row in rows
            ]
        ), 200

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
                    "reportUrl": f"/workflow/{workflow_id}/report",
                }
            ),
            200,
        )
    except Exception as ex:
        return jsonify({"message": str(ex), "step": step.to_dict()}), 400


# vrátí informaci o live HTML reportu workflow
@app.route(
    "/workflow/<workflow_id>/report",
    methods=["GET"],
)
def get_workflow_report(workflow_id: str):

    workflow = load_workflow(
        workflow_id
    )

    if not workflow:
        return (
            jsonify(
                {
                    "message":
                    f"Workflow with ID:'{workflow_id}' "
                    "does not exist."
                }
            ),
            404,
        )

    try:
        report_path = Path(
            get_report_path(
                workflow.state
            )
        )

    except Exception as exc:
        return (
            jsonify(
                {
                    "message":
                    f"Could not resolve workflow report: {exc}"
                }
            ),
            500,
        )

    if not report_path.is_file():
        return (
            jsonify(
                {
                    "message":
                    "HTML report has not been created yet."
                }
            ),
            404,
        )

    return send_from_directory(
        report_path.parent,
        report_path.name,
    )


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
        try:
            default_params = _get_default_operation_params(
                op_id
            )
        except KeyError:
            raise KeyError(
                f"Operation {op_id} does not exist."
            )

        if op_params is None:
            op_params = {}

        if not isinstance(op_params, dict):
            raise ValueError(
                f"Parameters for operation '{op_id}' must be a dictionary."
            )

        merged_params = default_params.copy()
        merged_params.update(op_params)

        step_id = str(uuid.uuid4())

        new_step = WorkflowStep(
            step_id=step_id,
            operation_id=op_id,
            params=merged_params,
        )

        workflow.definition.steps.append(
            new_step
        )
        created_steps.append({"step_id": step_id, "operation_id": op_id})

    return {"created_steps": created_steps}


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
