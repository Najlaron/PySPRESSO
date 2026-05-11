from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

@dataclass
class ParameterDef:
    name: str
    type: str
    required: bool = False
    default: Any = None
    label: str = ""
    help: str = ""

class OperationTag(str, Enum):
    INITIALIZATION = "initialization"
    IO = "io"
    FILTER = "filter"
    TRANSFORMATION = "transformation"
    NORMALIZATION = "normalization"
    SCALING = "scaling"
    CORRECTION = "correction"
    STATISTICS = "statistics"
    VISUALIZATION = "visualization"
    CANDIDATE_SELECTION = "candidate_selection"
    MY_FUNCTION = "my_function"

@dataclass
class OperationDefinition:
    id: str
    label: str
    description: str
    func: Callable[..., Any]

    category_tags: list[OperationTag] = field(default_factory=list)
    parameter_schema: list[ParameterDef] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    
