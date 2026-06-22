import importlib
import pkgutil
import pyspresso_app.operations as operations


def autoload_operations() -> None:
    """
    SIMPLER LOCAL VERSION

    """
    for module_info in pkgutil.iter_modules(operations.__path__):
        if module_info.ispkg:
            continue

        module_name = f"pyspresso_app.operations.{module_info.name}"
        print("Importing:", module_name)
        importlib.import_module(module_name)
