@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title DFSignal Launcher

call :detect_runtime
if not defined DF_RUN goto no_runtime

goto menu

:detect_runtime
set "DF_RUN="
set "DF_RUNTIME="
if not exist "pyproject.toml" exit /b 0

rem A: prefer uv only when its project environment already exists.
rem This validation is intentionally non-mutating; setup is explicit in the menu.
where uv >nul 2>&1
if not errorlevel 1 if exist ".venv\Scripts\python.exe" (
    uv run --no-sync python -c "import dfsignal" >nul 2>&1
    if not errorlevel 1 (
        set "DF_RUN=uv run --no-sync python"
        set "DF_RUNTIME=uv"
        exit /b 0
    )
)

rem B/C: use a local interpreter directly; activation is not required.
for %%E in (".venv\Scripts\python.exe" "venv\Scripts\python.exe" "env\Scripts\python.exe") do (
    if not defined DF_RUN if exist "%%~E" (
        "%%~E" -c "import dfsignal" >nul 2>&1
        if not errorlevel 1 (
            set "DF_RUN="%%~E""
            if "%%~E"==".venv\Scripts\python.exe" set "DF_RUNTIME=.venv"
            if "%%~E"=="venv\Scripts\python.exe" set "DF_RUNTIME=venv"
            if "%%~E"=="env\Scripts\python.exe" set "DF_RUNTIME=env"
        )
    )
)

rem D: use the current Python only after validating the package import.
if not defined DF_RUN (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import dfsignal" >nul 2>&1
        if not errorlevel 1 (
            set "DF_RUN=python"
            set "DF_RUNTIME=Current Python"
        )
    )
)
exit /b 0


:no_runtime
cls
echo WHAT FAILED:
echo No working DFSignal Python environment was found.
echo.
echo POSSIBLE OPTIONS:
echo 1. Recommended: uv sync
echo 2. Alternative: python -m venv .venv
echo    .venv\Scripts\python.exe -m pip install -e .
echo 3. Check current Python: python --version
echo.
echo WHERE TO CHECK:
echo %CD%\pyproject.toml
pause
exit /b 1

:menu
cls
echo ========================================
echo          DFSignal Local Launcher
echo ========================================
echo Runtime: %DF_RUNTIME%
echo ========================================
echo  1. Open Dashboard
echo  2. Run Full Pipeline
echo  3. Refresh Public Data
echo  4. Check Internal Data
echo  5. Show DFSignal Status
echo  6. Run Steam Analysis
echo  7. Open Executive Summary
echo  8. Run Tests
echo  9. Open Output Folder
echo 10. Setup / Repair Environment
echo 11. Exit
echo.
set "choice="
set /p "choice=Select an option [1-11]: "

if "%choice%"=="1" goto dashboard
if "%choice%"=="2" goto full_pipeline
if "%choice%"=="3" goto public_pipeline
if "%choice%"=="4" goto internal_check
if "%choice%"=="5" goto status
if "%choice%"=="6" goto steam
if "%choice%"=="7" goto summary
if "%choice%"=="8" goto tests
if "%choice%"=="9" goto output
if "%choice%"=="10" goto setup
if "%choice%"=="11" goto done

echo Invalid selection.
pause
goto menu

:dashboard
cls
echo Opening DFSignal Dashboard...
%DF_RUN% -m dfsignal dashboard
if errorlevel 1 (
    echo.
    echo WHAT FAILED: The dashboard did not start.
    echo POSSIBLE REASON: Streamlit or the local package is unavailable.
    echo RECOMMENDED ACTION: Select Setup / Repair Environment, then retry.
    echo LOG / COMMAND USED: %DF_RUN% -m dfsignal dashboard
)
pause
goto menu

:full_pipeline
cls
echo Running DFSignal pipeline...
%DF_RUN% -m dfsignal run-all --no-fetch-external
if errorlevel 1 echo WHAT FAILED: Full pipeline failed. Check outputs\latest_recap.md and the command output above.
pause
goto menu

:public_pipeline
cls
echo Refreshing public data and running DFSignal pipeline...
%DF_RUN% -m dfsignal run-all
if errorlevel 1 echo WHAT FAILED: Public-data pipeline failed. Check source-health artifacts and the command output above.
pause
goto menu

:internal_check
cls
echo Checking internal demand data...
%DF_RUN% -m dfsignal internal-check
if errorlevel 1 echo WHAT FAILED: Internal data validation failed or is not configured. No internal data was changed.
pause
goto menu

:status
cls
echo Checking DFSignal status...
%DF_RUN% -m dfsignal status
if errorlevel 1 echo WHAT FAILED: Status could not be read. Check the command output above.
pause
goto menu

:steam
cls
echo Running Steam Hardware Survey analysis...
%DF_RUN% -m dfsignal steam-real --no-fetch-external
if errorlevel 1 echo WHAT FAILED: Steam analysis failed. Check validation output above.
pause
goto menu

:summary
cls
echo Generating and opening the latest executive summary...
%DF_RUN% -m dfsignal summary --open-summary
if errorlevel 1 echo WHAT FAILED: Summary generation failed. Check outputs and the command output above.
pause
goto menu

:tests
cls
echo Running DFSignal tests...
%DF_RUN% -m pytest
if errorlevel 1 echo WHAT FAILED: Tests failed. Review the test output above.
pause
goto menu

:output
cls
echo Opening the DFSignal output folder...
explorer "%CD%\outputs"
goto menu

:setup
cls
echo Setup / Repair was selected. No setup runs automatically.
where uv >nul 2>&1
if not errorlevel 1 (
    echo Running uv sync...
    uv sync
) else (
    echo uv was not found. Creating or repairing .venv with current Python...
    where python >nul 2>&1
    if errorlevel 1 (
        echo WHAT FAILED: No Python command is available for setup.
        echo RECOMMENDED ACTION: Install Python or install uv, then retry.
        pause
        goto menu
    )
    python -m venv .venv
    if errorlevel 1 goto setup_failed
    ".venv\Scripts\python.exe" -m pip install -e .
)
if errorlevel 1 goto setup_failed
call :detect_runtime
if not defined DF_RUN goto setup_failed
echo Setup completed. Runtime: %DF_RUNTIME%
pause
goto menu

:setup_failed
echo.
echo WHAT FAILED: Setup / Repair did not complete.
echo POSSIBLE REASON: Package installation or environment creation failed.
echo WHERE TO CHECK: The command output above and pyproject.toml.
pause
goto menu

:done
endlocal
exit /b 0
