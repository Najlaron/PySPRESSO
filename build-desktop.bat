@echo off
REM ============================================================
REM  PySPRESSO - builds a standalone Windows desktop app
REM  (bundles the frontend + Python backend into one .exe, no
REM   Docker/Node/Python required for the end user afterwards)
REM ============================================================
setlocal

pushd "%~dp0"

echo.
echo [1/4] Installing frontend dependencies...
pushd frontend
call npm install
if errorlevel 1 goto :error
popd

echo.
echo [2/4] Building frontend (static assets)...
pushd frontend
call npm run build
if errorlevel 1 goto :error
popd

echo.
echo [3/4] Installing backend build dependencies...
pushd pyspresso
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python -m pip install pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 goto :error

echo.
echo [4/4] Building PySPRESSO.exe with PyInstaller...
python -m PyInstaller --noconfirm PySPRESSO.spec
if errorlevel 1 goto :error
popd

echo.
echo ============================================================
echo  Build complete!
echo  The app is in: pyspresso\dist\PySPRESSO\PySPRESSO.exe
echo  Copy the whole "PySPRESSO" folder to distribute/run it.
echo ============================================================
popd
pause
exit /b 0

:error
echo.
echo Build FAILED. See the messages above for details.
popd
pause
exit /b 1
