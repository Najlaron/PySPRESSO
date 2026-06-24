from typing import Any, Callable
from pyspresso_app.core.operation_models import (
    OperationDefinition,
    ParameterDef,
    OperationTag,
)

OPERATIONS: dict[str, OperationDefinition] = {}


def register_operation(
    *,
    id: str,
    label: str,
    description: str,
    citation: str = "",
    category_tags: list[OperationTag] | None = None,
    parameter_schema: list[ParameterDef] | None = None,
    requires: list[str] | None = None,
    produces: list[str] | None = None,
):
    def decorator(func: Callable[..., Any]):
        if id in OPERATIONS:
            raise ValueError(f"Operation '{id}' is already registered.")

        OPERATIONS[id] = OperationDefinition(
            id=id,
            label=label,
            description=description,
            citation=citation,
            func=func,
            category_tags=category_tags or [],
            parameter_schema=parameter_schema or [],
            requires=requires or [],
            produces=produces or [],
        )
        return func

    return decorator


def get_operation(operation_id: str) -> OperationDefinition:
    if operation_id not in OPERATIONS:
        raise KeyError(f"Unknown operation: {operation_id}")
    return OPERATIONS[operation_id]


def list_operations() -> list[OperationDefinition]:
    return list(OPERATIONS.values())


def operation_to_dict(op: OperationDefinition) -> dict:
    return {
        "id": op.id,
        "label": op.label,
        "description": op.description,
        "citation": op.citation,
        "category_tags": [tag.value for tag in op.category_tags],
        "parameter_schema": [
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


def list_operations_for_frontend() -> list[dict]:
    return [operation_to_dict(op) for op in OPERATIONS.values()]
