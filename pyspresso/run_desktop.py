"""PyInstaller entry point for the PySPRESSO desktop build.

Build with:  pyinstaller PySPRESSO.spec
(see build-desktop.bat at the repository root for the full build process)
"""

from pyspresso_app.desktop import main

if __name__ == "__main__":
    main()
