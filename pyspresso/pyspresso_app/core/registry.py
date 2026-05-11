from typing import Any, Callable
from core.operation_models import OperationDefinition, ParameterDef, OperationTag

OPERATIONS: dict[str, OperationDefinition] = {}


def register_operation(
    *,
    id: str,
    label: str,
    description: str,
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