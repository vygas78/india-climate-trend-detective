@echo off
setlocal
cd /d "%~dp0"
title EarthTrend Explorer

rem --- Find Python -------------------------------------------------------
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto no_python

rem --- Check Tkinter (comes with Python) -----------------------------------
%PY% -c "import tkinter" >nul 2>nul
if errorlevel 1 goto no_tk

rem --- Check / install required packages -----------------------------------
%PY% -c "import pandas, numpy, scipy, matplotlib" >nul 2>nul
if not errorlevel 1 goto run_app

echo Installing required packages. This needs internet and happens only once.
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed

%PY% -c "import pandas, numpy, scipy, matplotlib" >nul 2>nul
if errorlevel 1 goto pip_failed

:run_app
%PY% earthtrend_explorer.py
if errorlevel 1 (
    echo.
    echo EarthTrend Explorer closed with an error. Please read the message above.
    pause
)
goto end

:no_python
echo Python was not found on this computer.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" in the installer, then run this file again.
pause
goto end

:no_tk
echo Python was found, but Tkinter is missing.
echo Re-run the Python installer, choose Modify, and tick "tcl/tk and IDLE".
pause
goto end

:pip_failed
echo.
echo Package installation failed. Check your internet connection and try again.
echo You can also open Command Prompt in this folder and run:
echo     py -m pip install -r requirements.txt
pause

:end
endlocal
