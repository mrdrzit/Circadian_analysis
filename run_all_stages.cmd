@echo off
echo Running all Stages inside circadipy environment...

echo [STEP 1] Running Stage 1 inside circadipy environment...

"C:\Users\uzuna\miniforge3\envs\circadipy_env\python.exe" -m src.stage_1.run_stage_1
if errorlevel 1 goto :error

echo.
echo [STEP 2] Running Stage 2 inside circadipy environment...

"C:\Users\uzuna\miniforge3\envs\circadipy_env\python.exe" -m src.stage_2.run_stage_2
if errorlevel 1 goto :error

echo.
echo [STEP 3] Running Stage 3 inside circadipy environment...

"C:\Users\uzuna\miniforge3\envs\circadipy_env\python.exe" -m src.stage_3.run_stage_3
if errorlevel 1 goto :error

echo.
echo [OK] All stages completed.
pause
exit /b 0

:error
echo.
echo [ERROR] Pipeline stopped because a stage failed.
pause
exit /b 1
