from pyspresso_app.core.autoload import autoload_operations

_initialized = False


def initialize() -> None:
    global _initialized

    if _initialized:
        return

    autoload_operations()
    _initialized = True
