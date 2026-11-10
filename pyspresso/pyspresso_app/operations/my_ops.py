import numpy as np
from pyspresso_app.core.registry import register_operation
from pyspresso_app.core.operation_models import OperationTag, ParameterDef
from pyspresso_app.core.workflow_models import WorkflowState
from pyspresso_app.core.html_reporter import (
    add_text,
    add_table,
    add_figure,
)


##################################################################################
## EXAMPLE HOW TO PROPERLY REGISTER YOUR OWN OPERATION TO ADD IT INTO PySPRESSO ##
##################################################################################
