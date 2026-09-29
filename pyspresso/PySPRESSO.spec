# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for the PySPRESSO desktop application.

Build with (from the pyspresso/ folder, or via build-desktop.bat at the repo root):
    pyinstaller --noconfirm PySPRESSO.spec
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH)

# Operation modules are imported dynamically by name (autoload.py), so
# PyInstaller's static analysis can't discover them on its own.
hidden_imports = collect_submodules("pyspresso_app.operations")
hidden_imports += collect_submodules("pyspresso_app.core")

frontend_dist = project_root / "pyspresso_app" / "frontend_dist"
datas = []
if frontend_dist.is_dir():
    datas.append((str(frontend_dist), "pyspresso_app/frontend_dist"))

a = Analysis(
    ["run_desktop.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PySPRESSO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PySPRESSO",
)
